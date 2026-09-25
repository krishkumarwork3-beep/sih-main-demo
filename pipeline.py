"""
Per-camera worker process.

Runs in its OWN OS process (spawned by camera_manager.py) so that N cameras
scale across CPU cores instead of fighting the GIL — this is the Step 5
"one reusable worker per registry entry" model, not a per-camera script.

Responsibilities (all scoped to THIS camera only — nothing here compares
across cameras; that is reid_matcher.py's job, fed by the events this file
emits):
  - Step 7  Human & vehicle detection (YOLOv8)
  - Step 8  Multi-object tracking (BoT-SORT/ByteTrack, via custom_tracker.yaml)
  - Step 9  Virtual fence / line-crossing, IF this camera has a fence configured
  - Step 10 Face detection + recognition (InsightFace SCRFD + ArcFace),
            with best-shot-per-track persistence (unknown vs no_usable_face)
  - Step 11 ANPR (EasyOCR on vehicle crops)
  - Step 13 Night/low-light handling, tier 1 (brightness-gated histogram eq.)
  - Step 6  Camera health heartbeat (fps, blank-frame/obstruction check)
  - Step 18 Re-ID embedding emission — OSNet appearance vector per
            person-track to a shared, append-only event log that
            reid_matcher.py consumes to do the actual cross-camera matching.

Existing single-camera diagnostics from the original demo (ID-switch and
identity-swap detection) are preserved per-camera, unchanged in spirit.
"""
import argparse
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
SNAPSHOT_DIR = os.path.join(WEB_DIR, "snapshots")
TRACKER_CONFIG = os.path.join(BASE_DIR, "custom_tracker.yaml")
REID_EVENTS_PATH = os.path.join(WEB_DIR, "reid_events.jsonl")

PERSON_CLASS = 0
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck (COCO ids)

BOX_COLOR_NORMAL = (0, 200, 0)
BOX_COLOR_ALERT = (0, 0, 255)

FENCE_DEADZONE_PX = 8
CROSS_CONFIRM_FRAMES = 2
LOG_COOLDOWN_SEC = 4
IDENTITY_SIM_THRESH = 0.45
HEIGHT_RATIO_MIN = 0.7
HEIGHT_RATIO_MAX = 1.35
IDENTITY_LOG_COOLDOWN_SEC = 5
ID_SWITCH_DIST_PX = 120
ID_SWITCH_WINDOW_SEC = 3
REID_EMIT_INTERVAL_SEC = 1.0     # how often per track we emit a Step 18 embedding
NIGHT_BRIGHTNESS_THRESHOLD = 70  # mean gray value below this -> apply histeq (Step 13 tier 1)
HEALTH_UPDATE_EVERY_N_FRAMES = 8
BLANK_FRAME_STD_THRESHOLD = 4.0  # near-zero variance frame == obstructed/blank lens

# Windows can raise PermissionError on os.replace() if another process (here,
# camera_manager.py's run_merger loop, which polls every camera's health/
# alerts/detections files once a second) happens to have the destination
# file open for reading at the exact moment we try to swap it. This is a
# transient lock, not a real failure, so a short bounded retry clears it —
# on Linux os.replace() never raises this, so the loop just succeeds first try.
SAVE_JSON_MAX_RETRIES = 5
SAVE_JSON_RETRY_DELAY_SEC = 0.05

from face_engine import TrackFaceState, associate_faces_to_box, get_face_engine
from reid_embedder import cosine_similarity, get_reid_embedder

def resolve_device(requested):
    """--device auto picks CUDA when available and REFUSES to silently fall
    back to CPU without saying so — a GPU box quietly running YOLO on CPU
    looks like "it's just slow," not an error, so it's easy to miss.

    RTX 5060 note: it's a Blackwell (sm_120) GPU. torch.cuda.is_available()
    can return True while still not actually working, because a torch build
    whose bundled CUDA kernels predate sm_120 will report the GPU as present
    but throw a "no kernel image is available" (or similar) RuntimeError on
    the FIRST real op it tries to run on it — which for this script is a few
    seconds into model.track(), not at startup. That looks like the demo
    "hanging while loading". We do one tiny real op here, at startup, so that
    failure surfaces immediately with a clear message instead of mid-stream.
    """
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda") + 1  # force a real kernel launch now, not later
        except RuntimeError as exc:
            raise SystemExit(
                "torch.cuda.is_available() is True but a real CUDA op just failed: "
                f"{exc}\nThis is the classic RTX 5060 (Blackwell, sm_120) symptom — your "
                "installed torch build's CUDA kernels don't cover sm_120 yet, so it sees "
                "the GPU but can't actually run on it. Fix: reinstall torch from the cu128 "
                "(or newer) index, e.g.:\n"
                "  pip uninstall torch torchvision torchaudio\n"
                "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128\n"
                "(check https://pytorch.org/get-started/locally/ for the current index name — "
                "sm_120 support landed in the cu128 builds; cu121/cu124 builds do NOT have it)."
            ) from exc
        return "0"  # ultralytics device id for first CUDA GPU
    if requested == "cuda":
        raise SystemExit("--device cuda was requested but torch.cuda.is_available() "
                          "is False. Check your CUDA/driver install (see the RTX 5060 "
                          "Blackwell sm_120 note above) rather than silently running on CPU.")
    print("No CUDA GPU detected by torch — falling back to CPU. This will be slow "
          "for multi-camera inference.")
    return "cpu"


try:
    import easyocr
    OCR_READER = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
except Exception as exc:  # pragma: no cover
    print(f"EasyOCR unavailable ({exc}) — ANPR will be skipped.")
    OCR_READER = None


def cam_path(camera_id, name):
    return os.path.join(WEB_DIR, f"{name}_{camera_id}.json")


def frame_path(camera_id):
    return os.path.join(WEB_DIR, f"latest_frame_{camera_id}.jpg")


def health_path(camera_id):
    return os.path.join(WEB_DIR, f"health_{camera_id}.json")


def save_json(path, data, tail=None):
    if tail is not None:
        data = data[-tail:]
    tmp = path + f".tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
        return

    for attempt in range(SAVE_JSON_MAX_RETRIES):
        try:
            os.replace(tmp, path)
            return
        except (PermissionError, FileNotFoundError, OSError):
            if attempt == SAVE_JSON_MAX_RETRIES - 1:
                try:
                    with open(path, "w") as f:
                        json.dump(data, f, indent=2)
                    if os.path.exists(tmp):
                        os.remove(tmp)
                    return
                except Exception:
                    pass
            time.sleep(SAVE_JSON_RETRY_DELAY_SEC)


def side_of_line(point, a, b):
    return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])


def confirmed_side(point, a, b):
    cross = side_of_line(point, a, b)
    length = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
    if length == 0:
        return 0
    dist_px = cross / length
    if abs(dist_px) < FENCE_DEADZONE_PX:
        return 0
    return 1 if dist_px > 0 else -1


def compute_risk(off_hours):
    zone_violation = 1
    score = zone_violation * 30 + (10 if off_hours else 0)
    severity = "High" if score >= 40 else "Medium" if score >= 30 else "Low"
    return score, severity


def is_off_hours(config):
    window = config.get("allowed_time_window", {"start_hour": 6, "end_hour": 18})
    hour = datetime.now().hour
    return not (window["start_hour"] <= hour < window["end_hour"])


def draw_outlined_text(img, text, origin, scale=1.0, thickness=2,
                        text_color=(255, 255, 255), outline_color=(0, 0, 0),
                        outline_thickness=3):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, origin, font, scale, outline_color,
                thickness + outline_thickness, cv2.LINE_AA)
    cv2.putText(img, text, origin, font, scale, text_color, thickness, cv2.LINE_AA)


def read_plate(frame, box):
    if OCR_READER is None:
        return None
    x1, y1, x2, y2 = box
    h = y2 - y1
    crop = frame[y1 + int(h * 0.5):y2, x1:x2]
    if crop.size == 0:
        return None
    results = OCR_READER.readtext(crop)
    if not results:
        return None
    best = max(results, key=lambda r: r[2])
    return best[1] if best[2] > 0.4 else None


def person_crop(img, box):
    x1, y1, x2, y2 = box
    crop = img[max(y1, 0):y2, max(x1, 0):x2]
    if crop.size == 0:
        return None
    return crop


def maybe_apply_night_handling(frame):
    """Step 13, tier 1: visible-light, moderately dark -> histogram-equalize
    the luma channel. True darkness (tier 2) is left alone — there's no
    ambient signal left to recover, and pretending otherwise would just
    amplify sensor noise into false detections."""
    gray_mean = cv2.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))[0]
    if gray_mean >= NIGHT_BRIGHTNESS_THRESHOLD:
        return frame, False
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR), True


def emit_reid_event(camera_id, track_id, cls, embedding, height, zone, face_label):
    """Step 18: write an OSNet appearance embedding for this track to the
    shared event log. reid_matcher.py tails this file and does the actual
    cross-camera comparison — this process never looks at other cameras."""
    event = {
        "camera_id": camera_id,
        "track_id": int(track_id),
        "cls": int(cls),
        "zone": zone,
        "height": float(height),
        "embedding": embedding.flatten().tolist(),
        "face_label": face_label,
        "ts": time.time(),
    }
    with open(REID_EVENTS_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")


def write_health(camera_id, status, fps_actual, blank):
    save_json(health_path(camera_id), {
        "camera_id": camera_id,
        "status": status,
        "fps": round(fps_actual, 1),
        "blank_frame": blank,
        "last_update": time.time(),
    })


def run_camera(cam_cfg, config, args, sync_state=None):
    camera_id = cam_cfg["camera_id"]
    zone = cam_cfg.get("zone", "Unassigned")
    source_path = os.path.join(BASE_DIR, cam_cfg["source"])
    fence = config.get("fences", {}).get(camera_id)
    device = resolve_device(args.device)
    want_cuda = device != "cpu"
    model = YOLO(args.model)
    model.to(device if device == "cpu" else f"cuda:{device}")
    face_engine = get_face_engine(want_cuda=want_cuda)
    reid_embedder = get_reid_embedder(device="cuda" if want_cuda else "cpu")

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    save_json(cam_path(camera_id, "alerts"), [])
    save_json(cam_path(camera_id, "detections"), [])
    save_json(cam_path(camera_id, "faces"), [])
    write_health(camera_id, "starting", 0.0, False)

    prev_side, pending_side, pending_count = {}, {}, {}
    crossed_ids = set()
    face_last_logged, plate_last_logged = {}, {}
    plate_last_attempt, track_plates = {}, {}
    track_embed, track_height, swap_last_logged = {}, {}, {}
    track_face = {}
    track_frame_count = {}
    last_pos, ever_seen_ids = {}, set()
    reid_last_emit = {}
    cached_faces = []
    alerts, detections, faces_log = [], [], []
    frame_no = 0
    id_switch_log = os.path.join(BASE_DIR, f"id_switch_log_{camera_id}.txt")
    open(id_switch_log, "w").close()

    def open_capture():
        if source_path in ("0", 0):
            return cv2.VideoCapture(0)
        if not os.path.isfile(source_path):
            write_health(camera_id, "offline", 0.0, False)
            raise SystemExit(f"[{camera_id}] video file not found: {source_path}")
        cap = cv2.VideoCapture(source_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source_path)  # last-resort fallback to the default backend
        if not cap.isOpened():
            write_health(camera_id, "offline", 0.0, False)
            raise SystemExit(
                f"[{camera_id}] could not open source: {source_path}. If this is an .mp4 on Windows, "
                f"the usual cause is opencv-python's bundled FFMPEG build missing the H.264 decoder.")
        # Verify first frame can be read
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            cap.release()
            write_health(camera_id, "offline", 0.0, False)
            raise SystemExit(f"[{camera_id}] video opened but cannot read frames: {source_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return cap

    cap = open_capture()
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[{camera_id}] ========== VIDEO PROPERTIES ==========")
    print(f"[{camera_id}] Source: {source_path}")
    print(f"[{camera_id}] Duration: {total_frames / fps:.2f} seconds")
    print(f"[{camera_id}] Frames: {int(total_frames)}")
    print(f"[{camera_id}] FPS: {fps:.1f}")
    print(f"[{camera_id}] Resolution: {width}x{height}")
    print(f"[{camera_id}] =====================================")

    print(f"[{camera_id}] running ({cam_cfg['location_name']}, zone={zone}) on device={device} "
          f"({'GPU: ' + torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'}).")
    fps_window_start = time.time()
    fps_window_frames = 0

    if sync_state is not None:
        start_event, frame_counter, frame_cond, report_queue, stop_event = sync_state
        report_queue.put(("ready", camera_id, None))
        print(f"[{camera_id}] ready, waiting for synchronization start signal...")
        start_event.wait()

    stream_ended_normally = False
    try:
        while True:
            if sync_state is not None:
                with frame_cond:
                    while frame_counter.value < frame_no and not stop_event.is_set():
                        frame_cond.wait(timeout=0.2)
                if stop_event.is_set():
                    break

            ok, frame = cap.read()
            if not ok:
                if args.loop and source_path not in ("0", 0):
                    print(f"[{camera_id}] reached end of video, looping back to start...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if not ok:
                        cap.release()
                        cap = open_capture()
                        ok, frame = cap.read()
                    if not ok:
                        write_health(camera_id, "offline", 0.0, False)
                        raise SystemExit(f"[{camera_id}] Failed to read frame after looping")
                    write_health(camera_id, "healthy", fps, False)
                else:
                    stream_ended_normally = True
                    write_health(camera_id, "stream_ended", 0.0, False)
                    print(f"[{camera_id}] end of stream.")
                    if sync_state is not None:
                        report_queue.put(("ended", camera_id, frame_no))
                    break

            frame, night_mode = maybe_apply_night_handling(frame)

            results = model.track(
                frame, persist=True, tracker=TRACKER_CONFIG,
                classes=list({PERSON_CLASS} | VEHICLE_CLASSES),
                imgsz=args.imgsz, device=device, verbose=False,
            )[0]

            annotated = frame.copy()
            off_hours = args.force_off_hours or is_off_hours(config)
            frame_no += 1
            fps_window_frames += 1

            # --- Step 6: health heartbeat ---
            if frame_no % HEALTH_UPDATE_EVERY_N_FRAMES == 0:
                now = time.time()
                elapsed = max(now - fps_window_start, 1e-6)
                fps_actual = fps_window_frames / elapsed
                blank = float(np.std(frame)) < BLANK_FRAME_STD_THRESHOLD
                write_health(camera_id, "degraded" if blank else "healthy", fps_actual, blank)
                fps_window_start, fps_window_frames = now, 0

            has_persons = False
            if results.boxes.id is not None and results.boxes.cls is not None:
                clss_check = results.boxes.cls.cpu().numpy().astype(int)
                has_persons = any(c == PERSON_CLASS for c in clss_check)

            frame_faces = []
            if face_engine is not None and has_persons:
                if frame_no % 3 == 0 or not cached_faces:
                    try:
                        cached_faces = face_engine.detect(frame)
                    except Exception:
                        cached_faces = []
                frame_faces = cached_faces
            elif not has_persons:
                cached_faces = []

            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                ids = results.boxes.id.cpu().numpy().astype(int)
                clss = results.boxes.cls.cpu().numpy().astype(int)
                confs = results.boxes.conf.cpu().numpy()
                current_ids = set(ids.tolist())

                for box, track_id, cls, conf in zip(boxes, ids, clss, confs):
                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    label = "person" if cls == PERSON_CLASS else "vehicle"

                    if track_id not in ever_seen_ids:
                        ever_seen_ids.add(track_id)
                        best_match = None
                        for old_id, (ox, oy, ocls, oframe, oconf) in last_pos.items():
                            if old_id in current_ids or old_id == track_id:
                                continue
                            age_sec = (frame_no - oframe) / max(fps, 1)
                            if age_sec > ID_SWITCH_WINDOW_SEC or ocls != cls:
                                continue
                            dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
                            if dist <= ID_SWITCH_DIST_PX and (best_match is None or dist < best_match[1]):
                                best_match = (old_id, dist, age_sec, oconf)
                        if best_match:
                            old_id, dist, age_sec, oconf = best_match
                            msg = (f"[{camera_id}][frame {frame_no}] possible ID switch: #{old_id} "
                                   f"(conf {oconf:.2f}) vanished {age_sec:.2f}s ago -> new #{track_id} "
                                   f"(conf {conf:.2f}) appeared {dist:.0f}px away, class={label}")
                            print(msg)
                            with open(id_switch_log, "a") as f:
                                f.write(msg + "\n")

                    if cls == PERSON_CLASS:
                        track_frame_count[track_id] = track_frame_count.get(track_id, 0) + 1
                        t_frames = track_frame_count[track_id]

                        state = track_face.setdefault(track_id, TrackFaceState())
                        associated = associate_faces_to_box(frame_faces, (x1, y1, x2, y2))
                        if associated is not None and face_engine is not None:
                            state.update_from_detection(associated, face_engine.gallery)
                            bx1, by1, bx2, by2 = [int(v) for v in associated["bbox"]]
                            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (255, 200, 0), 1)
                            draw_outlined_text(annotated, state.label,
                                               (bx1, max(by1 - 8, 12)),
                                               scale=0.5, text_color=(255, 200, 0))
                        else:
                            draw_outlined_text(annotated, state.label,
                                               (x1, y2 + 16),
                                               scale=0.5, text_color=(255, 200, 0))

                        now = time.time()
                        if t_frames >= 2 and (track_id not in face_last_logged or now - face_last_logged[track_id] > LOG_COOLDOWN_SEC):
                            rec = {"camera_id": camera_id, "zone": zone,
                                   "timestamp": datetime.now().isoformat(timespec="seconds"),
                                   "track_id": int(track_id),
                                   "face_status": state.label,
                                   "label": state.label}
                            faces_log.append(rec)
                            save_json(cam_path(camera_id, "faces"), faces_log, tail=40)
                            detections.append({**rec, "type": "person", "detail": f"person #{track_id}"})
                            save_json(cam_path(camera_id, "detections"), detections, tail=30)
                            face_last_logged[track_id] = now

                        crop = person_crop(frame, (x1, y1, x2, y2))
                        embedding = reid_embedder.embed(crop) if (reid_embedder is not None and crop is not None) else None
                        height_px = y2 - y1
                        appearance_changed = height_changed = False
                        sim = height_ratio = None

                        if embedding is not None:
                            prev_emb = track_embed.get(track_id)
                            if prev_emb is not None:
                                sim = cosine_similarity(prev_emb, embedding)
                                appearance_changed = sim < IDENTITY_SIM_THRESH
                        prev_height = track_height.get(track_id)
                        if prev_height and prev_height > 0:
                            height_ratio = height_px / prev_height
                            height_changed = not (HEIGHT_RATIO_MIN <= height_ratio <= HEIGHT_RATIO_MAX)

                        if appearance_changed:
                            was_crossed = track_id in crossed_ids
                            crossed_ids.discard(track_id)
                            for d in (prev_side, pending_side, pending_count, plate_last_logged):
                                d.pop(track_id, None)

                        if appearance_changed or height_changed:
                            last_logged = swap_last_logged.get(track_id, 0)
                            if time.time() - last_logged > IDENTITY_LOG_COOLDOWN_SEC:
                                reasons = []
                                if appearance_changed:
                                    reasons.append(f"appearance similarity {sim:.2f}")
                                if height_changed:
                                    reasons.append(f"height ratio {height_ratio:.2f}x")
                                msg = (f"[{camera_id}][frame {frame_no}] possible IDENTITY SWAP: #{track_id} "
                                       f"({', '.join(reasons)})")
                                print(msg)
                                with open(id_switch_log, "a") as f:
                                    f.write(msg + "\n")
                                swap_last_logged[track_id] = time.time()

                        if embedding is not None:
                            track_embed[track_id] = embedding
                            if conf >= 0.22 and height_px >= 45 and t_frames >= 2:
                                now = time.time()
                                if now - reid_last_emit.get(track_id, 0) >= REID_EMIT_INTERVAL_SEC:
                                    emit_reid_event(camera_id, track_id, cls, embedding, height_px, zone, state.label)
                                    reid_last_emit[track_id] = now
                        track_height[track_id] = height_px

                        if fence:
                            a, b = fence[0], fence[1]
                            sign = confirmed_side((cx, cy), a, b)
                            stable = prev_side.get(track_id)
                            if sign == 0:
                                pass
                            elif stable is None:
                                prev_side[track_id] = sign
                            elif sign == stable:
                                pending_count[track_id] = 0
                            else:
                                if pending_side.get(track_id) == sign:
                                    pending_count[track_id] = pending_count.get(track_id, 0) + 1
                                else:
                                    pending_side[track_id] = sign
                                    pending_count[track_id] = 1
                                if pending_count[track_id] >= CROSS_CONFIRM_FRAMES:
                                    direction = "inbound" if sign > 0 else "outbound"
                                    score, severity = compute_risk(off_hours)
                                    snap_name = f"{camera_id}_track{track_id}_{int(time.time())}.jpg"
                                    cv2.imwrite(os.path.join(SNAPSHOT_DIR, snap_name), frame)
                                    alert = {"camera_id": camera_id, "zone": zone,
                                             "timestamp": datetime.now().isoformat(timespec="seconds"),
                                             "track_id": int(track_id), "event_type": "virtual_fence_crossing",
                                             "direction": direction, "off_hours": off_hours,
                                             "risk_score": score, "severity": severity,
                                             "snapshot": f"snapshots/{snap_name}"}
                                    alerts.append(alert)
                                    save_json(cam_path(camera_id, "alerts"), alerts, tail=50)
                                    print(f"[{camera_id}] ALERT: {alert}")
                                    crossed_ids.add(track_id)
                                    prev_side[track_id] = sign
                                    pending_count[track_id] = 0

                    box_color = BOX_COLOR_ALERT if track_id in crossed_ids else BOX_COLOR_NORMAL
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                    draw_outlined_text(annotated, f"{label} #{track_id}", (x1, max(y1 - 8, 12)))

                    if cls in VEHICLE_CLASSES:
                        now = time.time()
                        if track_id not in plate_last_attempt or (now - plate_last_attempt[track_id] > 2.0):
                            plate_last_attempt[track_id] = now
                            detected_plate = read_plate(frame, (x1, y1, x2, y2))
                            if detected_plate:
                                track_plates[track_id] = detected_plate
                        plate = track_plates.get(track_id)
                        if plate:
                            draw_outlined_text(annotated, f"plate: {plate}", (x1, y2 + 16), text_color=(0, 150, 255))
                            if track_id not in plate_last_logged or now - plate_last_logged[track_id] > LOG_COOLDOWN_SEC:
                                detections.append({"camera_id": camera_id, "zone": zone,
                                                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                                                    "type": "plate", "track_id": int(track_id), "detail": plate})
                                save_json(cam_path(camera_id, "detections"), detections, tail=30)
                                plate_last_logged[track_id] = now

                    last_pos[track_id] = (cx, cy, cls, frame_no, float(conf))

            if fence:
                cv2.line(annotated, tuple(fence[0]), tuple(fence[1]), (0, 0, 255), 2)
            draw_outlined_text(annotated, f"{camera_id} · {cam_cfg['location_name']}" + (" · NIGHT MODE" if night_mode else ""),
                                (16, 34), scale=0.8, text_color=(255, 255, 0) if night_mode else (255, 255, 255))

            cv2.imwrite(frame_path(camera_id), annotated)

            if sync_state is not None:
                report_queue.put(("done", camera_id, frame_no))
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"[{camera_id}] FATAL ERROR: {exc}")
        import traceback
        traceback.print_exc()
        write_health(camera_id, "offline", 0.0, False)
        raise
    finally:
        cap.release()
        if stream_ended_normally:
            write_health(camera_id, "stream_ended", 0.0, False)
        else:
            write_health(camera_id, "offline", 0.0, False)
        if sync_state is not None:
            try:
                report_queue.put(("ended", camera_id, frame_no))
            except Exception:
                pass