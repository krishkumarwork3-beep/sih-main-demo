"""
Multi-Camera Stream Manager — Step 5.

Reads the Camera Registry (Step 4, camera_registry.json) and spins up one
worker process per entry automatically — adding a camera to production is
a new registry row, not new code. Also starts the cross-camera Re-ID
matcher (Steps 19/20/21) and a lightweight merger that combines every
camera's alerts/detections/health into the single feed the dashboard polls.

Usage:
    python camera_manager.py
    python camera_manager.py --imgsz 1280 --model yolov8s.pt
    python camera_manager.py --no-loop          # stop each camera at end of its clip
    python camera_manager.py --force-off-hours  # demo severity escalation on demand
"""
import argparse
import json
import multiprocessing as mp
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
REGISTRY_PATH = os.path.join(BASE_DIR, "camera_registry.json")
REID_EVENTS_PATH = os.path.join(WEB_DIR, "reid_events.jsonl")

MERGE_INTERVAL_SEC = 1.0


def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


SAVE_JSON_MAX_RETRIES = 5
SAVE_JSON_RETRY_DELAY_SEC = 0.05


def save_json(path, data):
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


def run_merger(registry):
    """Combines every camera's per-camera files into the merged files the
    dashboard actually polls, and turns each camera's heartbeat timestamp
    into a live status (Step 6) — including flagging a camera as `offline`
    if its heartbeat has simply gone silent, distinct from a demo clip that
    ended on purpose (`stream_ended` / `stream_ended_looping`)."""
    from pipeline import cam_path, health_path  # local import: only merger needs these

    while True:
        merged_alerts, merged_detections, merged_health, merged_faces = [], [], [], []
        for cam in registry:
            cid = cam["camera_id"]
            merged_alerts.extend(load_json(cam_path(cid, "alerts"), []))
            merged_detections.extend(load_json(cam_path(cid, "detections"), []))
            merged_faces.extend(load_json(cam_path(cid, "faces"), []))

            h = load_json(health_path(cid), None)
            if h is None:
                merged_health.append({"camera_id": cid, "status": "starting", "fps": 0, "last_update": None})
                continue
            if h["status"] not in ("stream_ended", "stream_ended_looping") and \
                    time.time() - h["last_update"] > 15:
                h = {**h, "status": "offline"}
            h["location_name"] = cam["location_name"]
            h["zone"] = cam["zone"]
            merged_health.append(h)

        merged_alerts.sort(key=lambda a: a["timestamp"], reverse=True)
        merged_detections.sort(key=lambda d: d["timestamp"], reverse=True)
        merged_faces.sort(key=lambda d: d["timestamp"], reverse=True)
        save_json(os.path.join(WEB_DIR, "alerts.json"), merged_alerts[:50])
        save_json(os.path.join(WEB_DIR, "detections.json"), merged_detections[:30])
        save_json(os.path.join(WEB_DIR, "faces.json"), merged_faces[:40])
        save_json(os.path.join(WEB_DIR, "camera_health.json"), merged_health)
        time.sleep(MERGE_INTERVAL_SEC)


def run_sync_coordinator(camera_ids, start_event, frame_counter, frame_cond, report_queue, stop_event, target_fps=25.0, loop=False):
    """Coordinates all camera processes so they start at the exact same time
    and advance frame-by-frame in lockstep at target_fps."""
    camera_ids_set = set(camera_ids)
    ready = set()
    while len(ready) < len(camera_ids) and not stop_event.is_set():
        try:
            msg, cid, _ = report_queue.get(timeout=1.0)
            if msg == "ready":
                ready.add(cid)
        except Exception:
            continue
    if stop_event.is_set():
        return

    print(f"[sync-coordinator] All {len(ready)} camera processes initialized. Starting synchronous playback at {target_fps:.1f} fps...")
    start_event.set()

    active = set(camera_ids)
    f = 0
    frame_interval = 1.0 / max(target_fps, 1.0)

    while active and not stop_event.is_set():
        t0 = time.time()
        with frame_cond:
            frame_counter.value = f
            frame_cond.notify_all()

        done = set()
        while (done | (camera_ids_set - active)) < camera_ids_set and not stop_event.is_set():
            try:
                msg, cid, f_reported = report_queue.get(timeout=1.0)
                if msg == "done":
                    done.add(cid)
                elif msg == "loop":
                    done.add(cid)
                elif msg == "ended":
                    if not loop:
                        active.discard(cid)
                    done.add(cid)
            except Exception:
                # safety timeout: don't freeze indefinitely if a camera unexpectedly drops
                break

        elapsed = time.time() - t0
        rem = frame_interval - elapsed
        if rem > 0:
            time.sleep(rem)
        f += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--registry", default="camera_registry.json")
    parser.add_argument("--model", default="yolov8s.pt",
                         help="yolov8n.pt is the old default (kept for reference: fastest but weakest "
                              "boxes, which is a big source of dropped/duplicated ids). yolov8s.pt gives "
                              "noticeably more stable detections for basically free on an RTX 5060.")
    parser.add_argument("--imgsz", type=int, default=1280,
                         help="960 was the old default. 1280 catches a running person's box more "
                              "consistently frame-to-frame (fewer missed detections = fewer id switches) "
                              "and an RTX 5060 has plenty of headroom for it across 4 cameras.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                         help="auto (default): use CUDA if torch sees a GPU, else CPU with a warning. "
                              "cuda: require a GPU and error out if none is found, instead of silently "
                              "running on CPU. cpu: force CPU.")
    parser.add_argument("--force-off-hours", action="store_true")
    parser.add_argument("--loop", dest="loop", action="store_true",
                         help="Restart a camera's clip when it ends, for a continuous live demo "
                              "(default: OFF — each camera stops when its clip ends, and the whole "
                              "demo exits once every camera has stopped)")
    parser.add_argument("--sync", dest="sync", action="store_true", default=True,
                         help="Synchronize all camera streams frame-by-frame and pace at target fps (default: ON)")
    parser.add_argument("--no-sync", dest="sync", action="store_false",
                         help="Disable frame synchronization (cameras run independently)")
    parser.add_argument("--target-fps", type=float, default=25.0,
                         help="Target playback pace for synchronized streams (default: 25.0 fps)")
    parser.set_defaults(loop=False)
    args = parser.parse_args()

    from pipeline import run_camera  # deferred import so --help doesn't need torch/cv2
    from reid_matcher import run_reid_matcher

    registry = load_json(os.path.join(BASE_DIR, args.registry), [])
    if not registry:
        raise SystemExit(f"No cameras in {args.registry} — nothing to run.")
    config = load_json(os.path.join(BASE_DIR, args.config), {})

    os.makedirs(os.path.join(WEB_DIR, "snapshots"), exist_ok=True)
    print("Prefetching face (buffalo_l) and Re-ID (OSNet) weights if needed...")
    try:
        from face_engine import prefetch_face_models
        prefetch_face_models()
    except Exception as exc:
        print(f"InsightFace model prefetch failed ({exc}) — camera processes will retry and degrade if needed.")
    try:
        from reid_embedder import prefetch_reid_weights
        prefetch_reid_weights()
    except Exception as exc:
        print(f"OSNet weight prefetch failed ({exc}) — camera processes will retry and degrade if needed.")

    # Fresh run = fresh dashboard.
    open(REID_EVENTS_PATH, "w").close()
    for f in os.listdir(WEB_DIR):
        if f.startswith(("latest_frame_", "alerts_", "detections_", "health_", "faces_")):
            try:
                os.remove(os.path.join(WEB_DIR, f))
            except OSError:
                pass
    for name in ("faces.json", "alerts.json", "detections.json", "global_identities.json", "disambiguation_queue.json"):
        path = os.path.join(WEB_DIR, name)
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    fences = config.get("fences", {})
    enriched_registry = []
    for cam in registry:
        c = dict(cam)
        c["has_fence"] = cam["camera_id"] in fences
        c["fence_coords"] = fences.get(cam["camera_id"])
        c["fence_direction"] = "inbound" if cam["camera_id"] == "cam1" else "both"
        enriched_registry.append(c)
    save_json(os.path.join(WEB_DIR, "cameras.json"), enriched_registry)
    save_json(os.path.join(WEB_DIR, "config.json"), config)

    # Frame synchronization primitives
    sync_state = None
    coord_p = None
    if args.sync:
        start_event = mp.Event()
        stop_event = mp.Event()
        frame_counter = mp.Value("i", -1)
        frame_cond = mp.Condition(mp.Lock())
        report_queue = mp.Queue()
        sync_state = (start_event, frame_counter, frame_cond, report_queue, stop_event)

        cam_ids = [c["camera_id"] for c in registry]
        coord_p = mp.Process(
            target=run_sync_coordinator,
            args=(cam_ids, start_event, frame_counter, frame_cond, report_queue, stop_event, args.target_fps, args.loop),
            name="sync-coordinator",
            daemon=True
        )
        coord_p.start()

    camera_procs = []
    for cam in registry:
        p = mp.Process(target=run_camera, args=(cam, config, args, sync_state), name=f"cam-{cam['camera_id']}", daemon=True)
        p.start()
        camera_procs.append(p)

    reid_p = mp.Process(target=run_reid_matcher, args=(args, config), name="reid-matcher", daemon=True)
    reid_p.start()

    merger_p = mp.Process(target=run_merger, args=(registry,), name="dashboard-merger", daemon=True)
    merger_p.start()

    support_procs = [p for p in [coord_p, reid_p, merger_p] if p is not None]

    print(f"IBVAP multi-camera demo running — {len(registry)} camera(s). "
          f"Open web/index.html (serve the web/ folder, e.g. `python -m http.server` from it).")
    if args.loop:
        print("Looping enabled (--loop) — cameras restart their clip on end. Press Ctrl+C to stop.")
    else:
        print("Looping disabled (default) — the demo will stop on its own once every "
              "camera's clip has finished. Press Ctrl+C to stop early.")
    try:
        while any(p.is_alive() for p in camera_procs):
            time.sleep(1)
        if not args.loop:
            print("All camera clips have ended — stopping the demo (reid-matcher, merger, dashboard feeds).")
    except KeyboardInterrupt:
        pass
    finally:
        if args.sync and sync_state:
            sync_state[4].set()  # stop_event
        for p in camera_procs + support_procs:
            if p.is_alive():
                p.terminate()
        for p in camera_procs + support_procs:
            p.join(timeout=5)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()