# IBVAP Demo — v2 (multi-camera)

## What changed vs. the single-camera demo

The old `main.py` ran one hardcoded video through one detection/tracking/
fence loop. That's now Step 5's problem to solve, so it's been split up:

| File | Role |
|---|---|
| `camera_registry.json` | **Step 4.** One row per camera: id, location, source, zone. Add a camera by adding a row — nothing else changes. |
| `config.json` | Per-camera fence lines (`fences.<camera_id>`), shared time window, Re-ID tuning. |
| `pipeline.py` | **Steps 6, 7, 8, 9, 10, 11, 13, 18.** Everything that only needs to see *one* camera — detection, tracking, fence-crossing, face/plate, night handling, health heartbeat, and emitting a Re-ID embedding per track. Runs as its own OS process per camera. |
| `reid_matcher.py` | **Steps 19, 20, 21 (scoped).** Tails every camera's Re-ID events and does the actual cross-camera identity matching — this is the piece that's actually new. |
| `camera_manager.py` | **Step 5.** Reads the registry, spawns one `pipeline.py` process per camera + the matcher + a dashboard-file merger. Entry point for the whole demo now. |
| `fence_setup.py` | Same tool as before, now takes a `camera_id` and writes into that camera's fence entry. |
| `web/index.html` | Health strip, per-camera live-feed tiles (each with that camera's own fence status and face log), plus a separate "Cross-camera detection (all cameras)" aggregate. |

## Running it

```
pip install -r requirements.txt
pip uninstall -y onnxruntime
# insightface depends on the CPU `onnxruntime` wheel; uninstall it so
# onnxruntime-gpu==1.26.0 keeps CUDAExecutionProvider.
python camera_manager.py --device cuda   # all 4 cameras + Re-ID matcher
python -m http.server --directory web    # in another terminal
```
Open `http://localhost:8000`. Looping is off by default (each clip plays once).
Pass `--loop` for a continuous judging session.

Drop enrollment photos in `known_faces/` (optional). Filename stem is the
display name. An empty folder is fine.

**GPU:** every camera process auto-detects CUDA via `torch.cuda.is_available()`
and puts YOLO + EasyOCR on it — nothing here is hardcoded to CPU. On your
RTX 5060 setup this should just pick up the GPU automatically; each camera
prints which device it's actually running on at startup, so you can confirm
it. `--device cuda` makes that mandatory (errors out instead of silently
falling back to CPU if the GPU isn't seen — useful if a driver/CUDA build
issue is masquerading as "it's just slow"); `--device cpu` forces CPU.
Running 4 cameras at once is 4x the GPU memory/compute of the old single-
camera demo, so if you hit VRAM pressure, `--imgsz 640` or trimming
`camera_registry.json` down to 2 cameras for the live demo are the two
cheapest levers.

To set a fence line for a camera: `python fence_setup.py cam1`.

## How cross-camera Re-ID actually works here (Step 18-21)

1. Each camera's `pipeline.py` extracts an OSNet x1.0 appearance embedding
   from every person crop (body/clothing, not the face) and writes it, once
   a second per track, to a shared `web/reid_events.jsonl` (Step 18). Emissions
   are filtered to established tracks (>= 5 frames, conf >= 0.40, height >= 70px)
   preventing noise or false detections (such as on cam4's vehicle clip) from
   polluting the Re-ID space.
2. `reid_matcher.py` tails that file. When a *new* local track shows up on
   a camera, it compares that track's embedding against every recently-active
   identity using cosine similarity. It takes the top-2 nearest matches:
   - top-1 clears the similarity threshold (tuned to 0.50) with a clear margin
     over top-2 → merge into that global identity (Step 20's confident-merge case).
   - Once matched, the global feature vector is updated with an exponential moving
     average (`0.8 * old + 0.2 * new`), creating a stabilized multi-camera
     representation resilient to single-frame perspective shifts.
   - top-1 clears the threshold but margin is thin (< 0.03) → don't guess: create a
     provisional identity and push both candidates to the dashboard's
     disambiguation queue (Step 20's ambiguous case).
   - nothing clears the threshold → genuinely new global identity,
     `authorization_status: "unauthorized"` by default (Step 21).
3. The dashboard's **Cross-camera detection** section provides an isolated,
   dedicated view showing each global identity's camera trail (`cam1 → cam2 → cam3`),
   movement traversal timeline, disambiguation queue, and system-wide alerts.
   Per-camera fence status and face labels are kept completely segregated on each
   individual camera panel.

## Face detection + recognition (Step 10)

Haar cascade is gone (it isn't in OpenCV 5, and even when it worked it only
logged "a face was seen"). Each camera now runs InsightFace `buffalo_l`:
SCRFD detection + ArcFace (`w600k_r50`) embeddings on onnxruntime CUDA.

Drop enrollment photos in `known_faces/` (filename stem = display name). An
empty folder is valid — the detector still runs; every *usable* face is
labeled `unknown`.

Partial / briefly-visible faces are handled as the normal case, not an
edge case:
- **Best-shot-per-track:** keep the highest-quality crop so far (detector
  confidence × face size × landmark frontalness). Identity matching runs
  against that crop, not every frame.
- **Persistence:** a confirmed name (or confirmed-unknown) stays on the
  track for its lifetime. A later turned-away frame cannot flip a name
  back to `unknown`.
- **Confidence-aware matching:** a large frontal crop can match at a
  slightly lower cosine; a small/angled crop needs a higher score.
- **Two "no name" states:** `unknown` = a clean face was captured and is
  not in the gallery; `no_usable_face` = never got a crop good enough to
  attempt a confident match (hood up, range, angle).

The four provided clips have **no visible face by design** (hooded IR
subject on cam1–3; parked vehicle on cam4). `no_usable_face` / `unknown`
on them is the correct output, not a gap in the demo. Positive name
matches require enrollment photos with a visible face.

## What's a placeholder here, on purpose, vs. the plan

Being upfront about this matters more for a submission than pretending it's
further along than it is:

- **Adjacency graph (Step 19) = a flat time window, not geometry.** There's
  no GPS/transit-time graph; "candidate cameras" is just "any other camera
  active in the last `adjacency_window_sec` seconds." Real adjacency would
  narrow (and sanity-check) that candidate set.
- **No personnel database (Step 26) is wired up**, so `authorization_status`
  never flips to `authorized` for anyone in this build — that's honest
  behavior given there's nothing to authorize against, not a bug. A
  recognized `known_faces/` name is shown *alongside* unauthorized on the
  global identity, not used as an authorization grant.
- **Storage is JSON files, not PostgreSQL/FAISS.** Fine at 4-camera demo
  scale; the plan's own reasoning for why Postgres matters at real scale
  (relational queries, hundreds of cameras) still applies going forward.
- **Not touched at all:** posture/behavior signals (12), trajectory/rhythm/
  group analysis (14-17), the policy engine + zone-violation check (25-28),
  the full risk-scoring formula (32) — the demo still uses the original
  simplified `zone_violation*30 + off_hours*10` stand-in — blockchain
  evidence (33-38), dashboard admin panels (39-44), and everything in
  Phase 8 (real hardware/edge deployment, 45-51).

## Suggested framing for a submission demo

Given the plan's own scale (51 steps, PostgreSQL, Hyperledger Fabric, edge
deployment), no one should expect a from-scratch weekend build to be
feature-complete against it. What this build demonstrates concretely:
multi-camera ingestion scaling by registry row (Step 5), per-camera health
distinguishing "camera down" from "camera just went idle/off" (Step 6), and
— the centerpiece — a person tracked on one camera being re-identified on a
second camera with the ambiguous case handled honestly (operator queue)
rather than silently guessed (Steps 18-21). That maps cleanly onto Phases
1-4 of the plan; call out Phases 5-8 as the explicit next milestones rather
than letting a judge assume they're missing by accident.
