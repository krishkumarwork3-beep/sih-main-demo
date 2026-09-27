"""
Cross-camera identity matching — a scoped stand-in for Steps 19-21 of the
implementation plan.

What this is: a single process that tails the shared reid_events.jsonl
(one line per person-track OSNet appearance embedding, emitted by every
camera's pipeline.py — Step 18) and decides, per the plan's Step 20 logic,
whether a newly-appeared local track is:
  (a) an existing global identity re-appearing on a different camera -> merge
  (b) ambiguous between two candidates -> create a provisional identity and
      surface both candidates for operator disambiguation, per Step 20
  (c) genuinely new -> start a new global identity, authorization_status
      defaults to "unauthorized" (Step 21)

Matching uses cosine similarity on L2-normalized OSNet vectors. The
top-2 + margin decision rule is unchanged from the histogram-era design.

Adjacency (Step 19) is still a flat time window, not a GPS/transit-time
graph — called out so it isn't mistaken for the production design.
"""
import json
import os
import time

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
REID_EVENTS_PATH = os.path.join(WEB_DIR, "reid_events.jsonl")
GLOBAL_IDENTITIES_PATH = os.path.join(WEB_DIR, "global_identities.json")
DISAMBIGUATION_PATH = os.path.join(WEB_DIR, "disambiguation_queue.json")
REID_MAPPING_PATH = os.path.join(WEB_DIR, "reid_mapping.json")

POLL_INTERVAL_SEC = 1.0
STALE_IDENTITY_SEC = 300  # drop identities from active matching after 5 min idle


SAVE_JSON_MAX_RETRIES = 8
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


def event_embedding(ev):
    raw = ev.get("embedding", ev.get("hist"))
    if raw is None:
        return None
    vec = np.asarray(raw, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(vec))
    if n < 1e-8:
        return None
    return vec / n


def cosine(a, b):
    return float(np.dot(a, b))


def is_recognized_name(face_label):
    if not face_label:
        return False
    return face_label not in ("unknown", "insufficient_face_data")


def attach_face_label(g, face_label):
    if is_recognized_name(face_label) and not g.get("recognized_name"):
        g["recognized_name"] = face_label


def run_reid_matcher(args, config, global_reid_map=None):
    reid_cfg = config.get("reid", {})
    adjacency_window = reid_cfg.get("adjacency_window_sec", 45)
    sim_threshold = reid_cfg.get("similarity_threshold", 0.55)
    margin = reid_cfg.get("match_margin", 0.08)

    open(REID_EVENTS_PATH, "a").close()
    save_json(GLOBAL_IDENTITIES_PATH, [])
    save_json(DISAMBIGUATION_PATH, [])
    save_json(REID_MAPPING_PATH, {})

    global_identities = {}   # gid -> dict (feature kept as numpy)
    local_to_global = {}     # (camera_id, track_id) -> gid
    disambiguation_queue = []
    next_gid = [1]
    offset = 0

    def new_identity(camera_id, cls, zone, feature, height, ts, face_label):
        gid = f"G{next_gid[0]:04d}"
        next_gid[0] += 1
        g = {
            "global_id": gid,
            "cls": int(cls),
            "first_seen": ts,
            "last_seen": ts,
            "last_camera_id": camera_id,
            "cameras_seen": [camera_id],
            "authorization_status": "unauthorized",
            "recognized_name": None,
            "movement_history": [{"camera_id": camera_id, "zone": zone, "ts": ts}],
            "feature": feature,
            "height": height,
        }
        attach_face_label(g, face_label)
        global_identities[gid] = g
        return gid

    def touch(gid, camera_id, zone, feature, height, ts, log_move, face_label):
        g = global_identities[gid]
        g["last_seen"] = ts
        g["last_camera_id"] = camera_id
        if g.get("feature") is not None:
            updated = 0.8 * g["feature"] + 0.2 * feature
            n = np.linalg.norm(updated)
            if n > 1e-8:
                g["feature"] = updated / n
        else:
            g["feature"] = feature
        g["height"] = height
        attach_face_label(g, face_label)
        if camera_id not in g["cameras_seen"]:
            g["cameras_seen"].append(camera_id)
        if log_move and (not g["movement_history"] or g["movement_history"][-1]["camera_id"] != camera_id):
            g["movement_history"].append({"camera_id": camera_id, "zone": zone, "ts": ts})
            g["movement_history"] = g["movement_history"][-20:]

    print("[reid_matcher] running (cross-camera identity matching, OSNet cosine).")
    while True:
        try:
            with open(REID_EVENTS_PATH) as f:
                f.seek(offset)
                lines = f.readlines()
                offset = f.tell()
        except FileNotFoundError:
            time.sleep(POLL_INTERVAL_SEC)
            continue

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue

            feat = event_embedding(ev)
            if feat is None:
                continue
            key = (ev["camera_id"], ev["track_id"])
            ts = ev["ts"]
            face_label = ev.get("face_label") or "insufficient_face_data"

            if key in local_to_global and local_to_global[key] in global_identities:
                touch(local_to_global[key], ev["camera_id"], ev["zone"], feat, ev["height"], ts,
                      log_move=False, face_label=face_label)
                if global_reid_map is not None:
                    global_reid_map[f"{ev['camera_id']}_{ev['track_id']}"] = local_to_global[key]
                continue

            # Candidate matching: check recently-active identities of same class
            candidates = []
            recent_same_cls = []
            for gid, g in global_identities.items():
                if g["cls"] != ev["cls"]:
                    continue
                if ts - g["last_seen"] > adjacency_window:
                    continue
                recent_same_cls.append(gid)
                sim = cosine(g["feature"], feat)
                candidates.append((sim, gid))
            candidates.sort(key=lambda c: c[0], reverse=True)

            def assign_existing(gid):
                touch(gid, ev["camera_id"], ev["zone"], feat, ev["height"], ts,
                      log_move=True, face_label=face_label)
                local_to_global[key] = gid
                if global_reid_map is not None:
                    global_reid_map[f"{ev['camera_id']}_{ev['track_id']}"] = gid

            # Tracker ID switches / extra crops of the same person must not mint G0002
            # when there is already exactly one live identity of this class.
            if len(recent_same_cls) == 1:
                assign_existing(recent_same_cls[0])
                continue

            if candidates and candidates[0][0] >= sim_threshold:
                top1_sim, top1_gid = candidates[0]
                top2_sim = candidates[1][0] if len(candidates) > 1 else -1.0
                if top1_sim - top2_sim < margin:
                    disambiguation_queue.append({
                        "provisional_global_id": top1_gid,
                        "camera_id": ev["camera_id"],
                        "zone": ev["zone"],
                        "ts": ts,
                        "candidates": [
                            {"global_id": top1_gid, "similarity": round(float(top1_sim), 3)},
                            {"global_id": candidates[1][1], "similarity": round(float(top2_sim), 3)},
                        ],
                    })
                    disambiguation_queue = disambiguation_queue[-20:]
                assign_existing(top1_gid)
                continue

            gid = new_identity(ev["camera_id"], ev["cls"], ev["zone"], feat,
                               ev["height"], ts, face_label)
            local_to_global[key] = gid
            if global_reid_map is not None:
                global_reid_map[f"{ev['camera_id']}_{ev['track_id']}"] = gid

        now = time.time()
        for gid in list(global_identities.keys()):
            if now - global_identities[gid]["last_seen"] > STALE_IDENTITY_SEC:
                del global_identities[gid]

        out = []
        ranked = sorted(global_identities.values(), key=lambda g: g["last_seen"], reverse=True)
        for g in ranked:
            cameras = g.get("cameras_seen") or []
            dwell = float(g["last_seen"]) - float(g["first_seen"])
            # Hide one-frame ghosts (tracker fragments) when a real trail already exists.
            if len(cameras) <= 1 and dwell < 1.5:
                has_real = any(
                    (o["cls"] == g["cls"] and o["global_id"] != g["global_id"]
                     and ((o.get("cameras_seen") or []) and (
                         len(o.get("cameras_seen") or []) > 1
                         or (float(o["last_seen"]) - float(o["first_seen"])) >= 1.5)))
                    for o in ranked
                )
                if has_real:
                    continue
            g2 = {k: v for k, v in g.items() if k != "feature"}
            g2["cls_label"] = "person" if g["cls"] == 0 else "vehicle"
            out.append(g2)
        save_json(GLOBAL_IDENTITIES_PATH, out)
        save_json(DISAMBIGUATION_PATH, disambiguation_queue)
        save_json(REID_MAPPING_PATH, {f"{cid}_{tid}": gid for (cid, tid), gid in local_to_global.items()})

        time.sleep(POLL_INTERVAL_SEC)
