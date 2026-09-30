# IBVAP: Intelligent Border Video Analytics Platform

A multi-camera video surveillance system for the Sashastra Seema Bal (SSB) that combines real-time detection, cross-camera identity tracking, and policy-based authorization verification. Built for Smart India Hackathon 2024.

## Problem Statement

Border security operations generate continuous video across potentially hundreds of cameras, but most surveillance systems treat each camera as an isolated feed. A person moving through multiple camera zones appears as disconnected detections with no way to track their movement path or verify their authorization status across the perimeter. Manual monitoring cannot scale to this volume, and simple motion detection produces alert fatigue from routine activity.

IBVAP addresses this by building cross-camera identity continuity on top of per-camera detection. The system tracks the same person across multiple cameras, maintains an authorization state tied to that global identity (not frame-by-frame face matches), and applies zone-specific policy rules to generate actionable alerts. Authorization is determined once via face recognition and carried forward through subsequent cameras via appearance-based person re-identification, so a patrol member facing away from a camera or moving in poor lighting conditions is not falsely flagged.

The platform distinguishes between identity-tier evidence (policy violations, unauthorized entry points, route deviations) and behavior-tier observations (posture changes, movement patterns) in its risk scoring, ensuring behavioral signals only ever add context to an already-triggered authorization flag rather than generating alerts on their own.

## Architecture

```
┌──────────────┐
│   Cameras    │  (4 feeds: video.mp4, video2.mp4, video3.mp4, video4.mp4)
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Camera Registry & Stream Manager  (camera_manager.py)               │
│  - Reads camera_registry.json                                        │
│  - Spawns one pipeline.py process per camera                         │
│  - Spawns reid_matcher.py for cross-camera matching                  │
│  - Merges per-camera outputs → dashboard JSON files                  │
└──────────────────────────────────────┬───────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Per-Camera AI Pipeline  (pipeline.py, one process per camera)       │
│                                                                       │
│  Step 7:  YOLOv8 person/vehicle detection                            │
│  Step 8:  ByteTrack multi-object tracking                            │
│  Step 9:  Virtual fence line-crossing (if configured)                │
│  Step 10: InsightFace SCRFD + ArcFace face detection/recognition     │
│  Step 11: EasyOCR license plate recognition                          │
│  Step 13: Night-time histogram equalization (brightness-gated)       │
│  Step 6:  Health heartbeat (fps, blank-frame check)                  │
│  Step 18: OSNet Re-ID embedding emission → reid_events.jsonl         │
│                                                                       │
│  Outputs per camera:                                                 │
│    web/detections_camX.json                                          │
│    web/alerts_camX.json                                              │
│    web/faces_camX.json                                               │
│    web/health_camX.json                                              │
│    web/latest_frame_camX.jpg                                         │
└──────────────────────────────────────┬───────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Cross-Camera Re-ID Matcher  (reid_matcher.py)                       │
│                                                                       │
│  - Tails reid_events.jsonl (OSNet embeddings from all cameras)       │
│  - Top-2 cosine similarity matching with margin check                │
│  - Merges local track IDs → Global IDs (G0001, G0002, ...)           │
│  - Outputs: web/global_identities.json                               │
│              web/reid_mapping.json                                   │
│              web/disambiguation_queue.json (ambiguous cases)          │
└──────────────────────────────────────┬───────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Dashboard  (web/index.html)                                         │
│                                                                       │
│  - Per-camera panels: fence status, face log, live feed              │
│  - Cross-camera section: global identity trails, movement timeline   │
│  - Polls JSON files every second                                     │
│  - Static HTML + JavaScript (no backend server needed)               │
└──────────────────────────────────────────────────────────────────────┘
```

## Detection and Tracking

### YOLOv8 for Person and Vehicle Detection

Each camera feed runs YOLOv8 inference on every frame, detecting persons (class 0) and vehicles (classes 2, 3, 5, 7: car, motorcycle, bus, truck). We use the pretrained COCO weights with no custom training, which provides sufficient accuracy for border security scenarios where the object categories are already represented in the base dataset.

The model runs at configurable resolution (default 1280px) with GPU acceleration where available. For RTX 5060 Laptop GPU (Blackwell architecture, sm_120), the code includes explicit CUDA capability validation at startup to surface any kernel compatibility issues immediately rather than hanging mid-stream.

### ByteTrack for Persistent Local IDs

Detection alone provides no continuity across frames. We use ByteTrack for multi-object tracking, which assigns each detected person a persistent local track ID within that camera's view. ByteTrack was chosen over DeepSORT specifically because it maintains tracking continuity during ordinary occlusion and low-confidence detection moments, whereas DeepSORT discards low-confidence detections outright, breaking track persistence exactly when lighting conditions or partial occlusion drop detector confidence below threshold.

Track history (centroid positions, timestamps) is maintained in a rolling buffer per track, which serves as input to the virtual fence crossing check, movement pattern analysis, and Re-ID embedding emission.

##Virtual Fence Line-Crossing

An administrator defines the border line by clicking two points on a frame from the camera feed (`fence_setup.py`). The system stores these coordinates in `config.json` keyed by camera ID. During runtime, for each tracked object, the pipeline computes its centroid and evaluates the signed distance to the line using the geometric test:

```
d = (x2 - x1)(cy - y1) - (y2 - y1)(cx - x1)
```

A sign flip between consecutive frames indicates a crossing. The sign itself determines direction: positive to negative is inbound, negative to positive is outbound. Crossings are confirmed only after persisting for multiple frames (configurable via `CROSS_CONFIRM_FRAMES`) to filter single-frame noise. Each confirmed crossing generates an alert entry in `web/alerts_camX.json` with the track ID, direction, timestamp, and a snapshot.

This approach is deterministic and computationally cheap compared to a learned classifier, and it directly yields crossing direction without a separate inference step.

## Face Detection and Recognition

### InsightFace: SCRFD + ArcFace

The system uses InsightFace's `buffalo_l` model pack, which includes SCRFD for face detection and ArcFace (`w600k_r50`) for recognition. Both run via onnxruntime with CUDA acceleration, reusing the same GPU inference stack as YOLO rather than introducing a second framework dependency.

Face recognition operates on an enrollment model: administrators place images in the `known_faces/` directory (filename stem becomes the display name, e.g., `Krish.jpg`). On startup, the system detects and embeds each enrollment image into a searchable index. If the directory is empty, face detection still runs normally but all recognized faces are labeled as "unknown."

### Best-Shot-Per-Track Persistence

Real footage rarely provides frontal, well-lit faces in every frame. Instead of matching every frame independently (which produces flickering labels as face quality varies), the system maintains a best-shot-per-track: for each active track, it keeps the highest-quality face crop seen so far, scored by `detector_confidence × face_size × landmark_frontalness`. Identity matching runs against this best crop, not every frame.

Once a track's face is confidently matched to a name (or confirmed as "unknown"), that label persists for the track's lifetime. A later frame where the person turns their head away does not flip a confirmed name back to "unknown." This track-level persistence prevents label flicker and provides the stable authorization binding that cross-camera Re-ID depends on.

### Confidence-Aware Matching

A high-quality face crop (large, frontal, high detector confidence) can accept a slightly lower cosine similarity score as a match. A poor-quality crop (small, angled, low confidence) requires a higher similarity before accepting a match. This adaptive threshold reduces both false negatives on difficult angles and false positives on low-quality crops.

### Two "No Name" States

The system distinguishes between two distinct cases:

- `unknown`: A good-quality face was captured and successfully processed, but does not match anyone in the enrollment gallery (confirmed stranger).
- `no_usable_face`: Never obtained a face crop clean enough across the track's lifetime to attempt a confident match (face obscured, always at a bad angle, too far from camera, or in the case of the provided demo footage, wearing a hood).

These are stored and displayed as separate values. This distinction matters operationally: "unknown" means we have a clear face image but no enrollment match, whereas "no_usable_face" means the person's face was never clearly visible.

## Cross-Camera Re-ID and Global Identity

### OSNet Appearance Embeddings

ByteTrack IDs are local to a single camera. To recognize the same person across cameras, the system generates appearance embeddings using OSNet x1.0 (via `torchreid`), which captures clothing color, build, and silhouette from the full body crop. Unlike ArcFace, which requires a frontal face, OSNet works from behind or at an angle, making it suitable for cross-camera matching where face visibility cannot be guaranteed.

Each camera emits one OSNet embedding per track per second (configurable via `REID_EMIT_INTERVAL_SEC`) to a shared append-only event log (`web/reid_events.jsonl`), tagged with `camera_id`, `local_track_id`, `timestamp`, and the 512-dimensional embedding vector.

### Camera Adjacency Graph

Real deployment would include an adjacency graph (Step 19 in the implementation plan): a directed, transit-time-aware mapping of which cameras could plausibly see the same person within a given time window, based on walkable terrain rather than straight-line GPS distance. This narrows the candidate set for Re-ID comparison and improves precision.

The demo implementation uses a simplified time-window model: any camera active within the last 45 seconds (configurable via `reid.adjacency_window_sec` in `config.json`) is considered a candidate. Production deployment would replace this with surveyed, terrain-grounded adjacency edges.

### Top-2 Matching with Margin Check

The Re-ID matcher (`reid_matcher.py`) continuously tails `reid_events.jsonl`. When a new local track appears on any camera, it compares that track's embedding against all recently-active global identities using cosine similarity.

The comparison retrieves the top-2 nearest candidates:

1. If the top-1 candidate exceeds the similarity threshold (default 0.50) with a clear margin (default 0.03) over the top-2, the new local track is merged into that global identity.
2. If top-1 clears the threshold but the margin is thin, the match is ambiguous. A provisional identity is created and both candidates are surfaced to the dashboard's disambiguation queue (`web/disambiguation_queue.json`).
3. If no candidate clears the threshold, a genuinely new global identity is created with `authorization_status: "unauthorized"`.

Once matched, the global identity's feature vector is updated with an exponential moving average (`0.8 × old + 0.2 × new`), creating a multi-camera appearance representation that is more stable than any single detection.

### Local-to-Global Mapping

The matcher maintains a persistent mapping (`web/reid_mapping.json`) from `{camera_id}_{local_track_id}` to `global_identity_id` (e.g., `G0001`). This mapping is read by each camera's pipeline process to display the global ID on the annotated frame. When a confident Re-ID merge happens, a person who was "Track #1" on camera 1 and "Track #3" on camera 2 now shows as the same global track number derived from their shared global ID across both cameras.

## Risk and Intent Scoring

The full risk scoring engine (Step 32) is part of the platform design but not implemented in the demo codebase. The conceptual model is a two-tier formula:

```
identity_score = zone_violation×30 + no_traceable_origin×25 
                 + route_anomaly×15 + off_hours×10

context_multiplier = zone_sensitivity_factor × time_factor
  where zone_sensitivity_factor: 0.6 (levels 1-2) / 1.0 (level 3) / 1.4 (levels 4-5)
        time_factor: 0.7 (daytime) / 1.3 (night)

behavior_score_raw = trajectory_jagged×3 + stutter_gait×3 
                     + boundary_hesitation×3 + posture_signals×(2-4 each)

behavior_score = behavior_score_raw × context_multiplier

if identity_score == 0:
  behavior_contribution = min(behavior_score, 5)  # hard cap
else:
  behavior_contribution = behavior_score

risk_score = identity_score + behavior_contribution
```

The key design principle: identity-tier flags (zone violations, entry-point verification failures, route anomalies, off-hours activity) drive severity directly. Behavior-tier observations (posture, trajectory shape, movement rhythm) only add corroborating context when an identity-tier flag has already fired, and are capped at a low maximum contribution when evaluated alone, ensuring they cannot by themselves push a case into Medium/High/Critical severity bands.

## Cybersecurity

### Data Encryption and Access Control

The platform design includes AES-256 encryption for data at rest and JWT-based authentication with RS256 signatures for API access. Access tokens are short-lived (15-60 minutes) with a separate refresh token flow to limit exposure window. A server-side revocation list provides the ability to invalidate tokens before natural expiry, trading a small lookup cost for immediate revocation capability.

Role-based access control (RBAC) restricts administrative functions (zone configuration, personnel enrollment, policy updates) to designated roles. Every operator action, including alert overrides and policy changes, is logged with the operator's identity for audit trail purposes.

### Secure WebSocket Connections

The live alert feed and dashboard updates use secure WebSockets (`wss://`) with JWT authentication at handshake time. Connections are periodically re-validated for the life of the session, ensuring a revoked token cannot maintain an open connection indefinitely.

### Liveness Detection

Face enrollment and recognition include liveness checks (blink detection, depth verification via MediaPipe Face Mesh) to prevent photo-based spoofing attempts.

### Transport Security

All camera-to-server and edge-to-central connections use TLS/VPN with certificate-based device authentication. For edge deployments with intermittent connectivity, devices perform time synchronization (NTP or against the central server's clock) immediately upon reconnecting to prevent clock drift from corrupting timestamp-based event ordering.

## Blockchain Evidence Layer

### SHA-256 Hashing and Ledger Commits

Every evidence clip is hashed (SHA-256) at the time of capture. The hash is written as a transaction to Hyperledger Fabric, a permissioned blockchain chosen for its single-organization use case and freedom from public chain gas fees and throughput limits. Policy changes, user actions, and alert overrides are logged to the same ledger, creating a tamper-evident audit trail.

### Edge Write-Ahead Queue

Edge devices (for bandwidth-constrained or remote locations) maintain a local, durable write-ahead queue with two tiers:

1. **Tier 1 (metadata + hash)**: Small, never evicted. Contains `{event_id, sha256_hash, timestamp, event_type, risk_score, tier1_flags}`, the minimum record needed to prove an event occurred and wasn't altered.
2. **Tier 2 (raw evidence clip)**: Evictable under disk pressure. If local storage runs low during an extended connectivity outage, raw clips are pruned oldest-first, but the eviction itself is recorded as a Tier-1 event, so the chain-of-custody record permanently shows the clip existed, was hashed, and was later evicted due to capacity constraints.

Tier-1 queue depth is included in the camera health heartbeat. If it exceeds a conservative threshold, an operational alert recommends physical intervention (technician visit to manually offload the device) rather than silently deleting metadata to solve a storage problem.

### Chain-of-Custody States

Evidence in the dashboard is marked with one of three states:

- `pending_ledger_commit`: Event captured locally, not yet synced to the blockchain.
- `confirmed_on_chain`: Hash committed to Hyperledger Fabric.
- `evidence_unavailable (evicted)`: Tier-2 clip evicted due to storage pressure, but Tier-1 hash record remains on-chain.

This provides honest visibility into what is verifiable at any given moment, rather than claiming every clip is always available or silently substituting a missing clip with a stale copy.

### Merkle-Batch Anchoring

To reduce per-event blockchain overhead at scale, the system can accumulate events over a short time window and commit a single Merkle root per batch instead of one transaction per event. Each event's hash and Merkle proof (the sibling hashes needed to reconstruct the root) are stored in the database. Verification of any individual event still works the same way: recompute the event hash, walk the stored proof to the batch root, compare against the on-chain root. This reduces ledger transaction volume by approximately the batch size while preserving per-event tamper-evidence.

## IPFS Private Swarm

Evidence storage uses a private IPFS swarm: each edge device runs a local IPFS node, and clips are added to IPFS (producing a content identifier, CID) at the same time the SHA-256 hash is computed. Both values are written into the blockchain record. A central pinning node within SSB infrastructure pins every clip whose hash has reached `confirmed_on_chain` status, ensuring the clip remains retrievable past an individual device's local retention window.

This is a fully private network with a pre-shared swarm key, no public gateway, and no public pinning service. Nothing evidence-related touches the public IPFS network. Content addressing provides two benefits: retrieval no longer depends on one fixed file location (it can pull from any available replica), and byte-identical clips (e.g., static background frames captured by overlapping camera fields) are automatically deduplicated by IPFS without additional code.

## Generative AI: Incident Narratives and Shift Briefings

The alert schema (`tier1_flags`, `tier2_observations`, `risk_score`, timestamps) is structured data, not something a human can skim in five seconds during a shift handoff. A self-hosted LLM (Llama or Mistral class) generates plain-language summaries strictly grounded in the structured fields already computed by the detection and reasoning pipeline.

Generation is retrieval-augmented with access only to the alert payload, movement history, and camera health logs. The model is never given open-ended knowledge or permission to infer facts not present in the source record. It is a phrasing layer, not a reasoning layer. Every generated briefing carries model version provenance and links back to the underlying alert entry for verification. Generated text is never treated as evidence on its own.

Two caching layers sit in front of the generation call:

- **Prompt caching**: Fixed system instructions and field-grounding templates are cached at the inference layer, so only the alert-specific content is reprocessed per request.
- **Semantic caching**: Incoming alert payloads are embedded and checked against recent similar alerts. Near-duplicate patterns (same zone, same flag combination, same time-of-day) return the cached narrative instead of invoking fresh generation.

A semantic cache hit is labeled as such in the UI (`cached: true, source_event: <id>`) rather than presented as freshly generated.

## Agentic AI: Autonomous Alert Investigation

An operator confirming or dismissing an alert typically cross-checks multiple systems: camera health logs, adjacency graph, movement history, origin verification, evidence status. An agent with read-only access to a fixed tool registry can pre-assemble this cross-checking before the operator opens the alert.

The agent is given exactly one function per relevant system component: `get_camera_health(camera_id, window)`, `get_movement_history(global_identity_id)`, `get_origin_verification(global_identity_id)`, `get_evidence_status(event_id)`, `get_model_versions(event_id)`. On any Medium-or-higher-severity alert, the agent runs a fixed investigation routine (not open-ended autonomous planning) that calls the relevant subset of these tools and writes the results as a structured investigation packet attached to that alert in the dashboard feed.

The agent has **no write access** to alert overrides, policy tables, zone configuration, or personnel enrollment. It can only read and summarize. Every tool call the agent makes is logged to the blockchain audit trail, exactly like operator actions, so there is a reviewable record of what the agent checked and when.

This preserves the human-in-the-loop guarantee: the investigation is pre-assembled to save time, but the actual decision (confirmed, false positive, inconclusive) remains with a human operator.

## Dashboard

The web dashboard (`web/index.html`) is a static HTML page with JavaScript that polls JSON files in the `web/` directory every second. No backend server is required beyond serving the static files.

### Per-Camera Panels

Each camera has its own panel showing:

- Live feed (latest frame, updated continuously)
- Virtual fence status: last crossing event (if any), direction, timestamp, severity
- Face detections: name / "unknown" / "no_usable_face", track ID, timestamp
- Active track count and current FPS

These panels are independent: cam1's panel shows only cam1's fence crossings and face detections, not merged data from other cameras.

### Cross-Camera Detection Section

A separate, visually distinct section labeled "Cross-Camera Re-Identification & Movement Tracking" covers all cameras together:

- Global identities table: Global ID, authorization status, face identity (if recognized), cameras visited, last seen time
- Active cross-camera transit timeline: breadcrumb view showing `cam1 → cam2 → cam3` movement paths
- Disambiguation queue: ambiguous Re-ID matches awaiting operator confirmation

If a global identity has a recognized face label, that name is displayed alongside the global ID in this section.

### Camera Health Strip

A horizontal strip at the top shows health status for all cameras: healthy (green), degraded (yellow), offline (red), or stream_ended (blue). Clicking a camera in the health strip jumps to that camera's panel.

### System Logs Panel

The full audit trail (model version changes, operator overrides, policy updates) is displayed as a searchable table, sourced directly from the blockchain ledger for tamper-evidence. Includes adjacency graph maturity metrics (percentage of actively-queried edges that remain placeholder-only) and clock-drift fallback counts for operational visibility.

## Cost and Efficiency Techniques

### Motion-Gated Inference

A lightweight frame-differencing check runs ahead of YOLO. Only frames where pixel-level change exceeds a threshold are forwarded into the full detection/tracking/recognition pipeline. Frames below threshold are logged as "no activity" at near-zero cost. This cuts GPU and edge compute load in proportion to how much of the day a camera spends idle, which at a border post is often the majority of runtime.

### Quantized Edge Models

For edge deployments (bandwidth-constrained sites), INT8-quantized YOLOv8 and distilled ArcFace models replace full-precision central weights, reducing per-inference compute cost and lowering the edge hardware spec required. Accuracy is benchmarked against the full-precision version before rollout; sites where the quantized model does not meet the defined accuracy bound stay on full precision or are reclassified as central-bandwidth sites.

### Cascade LLM Routing

Most alerts land in Low/Medium severity bands and do not require deep reasoning. A small, locally-hosted model handles first-pass narrative generation and investigation for Medium-or-below alerts. Only High/Critical alerts are routed to the larger model. The routing decision reads directly from the risk score already computed, avoiding a separate learned router.

### Automated Retention Policy

Footage that was never flagged or was overridden as a confirmed false positive is deleted from cold storage once it passes a configured retention window. Anything attached to a confirmed alert, operator override, or investigation packet is exempted and retained per compliance requirements. The deletion job is scheduled during off-peak electricity windows and runs on spot/preemptible compute to reduce cost.

## Tech Stack

- **Detection & Tracking**: YOLOv8 (Ultralytics), ByteTrack
- **Face Recognition**: InsightFace (SCRFD + ArcFace),FAISS, onnxruntime-gpu
- **Person Re-ID**: OSNet x1.0 (torchreid)
- **OCR**: EasyOCR
- **Pose Estimation**: MediaPipe Pose, MediaPipe Face Mesh
- **Backend**: Python 3.10+, FastAPI (for production REST/WebSocket endpoints)
- **Database**: PostgreSQL, FAISS (replaced by pgvector/pgvectorscale in production)
- **Blockchain**: Hyperledger Fabric
- **Storage**: IPFS (private swarm), local filesystem for demo
- **Dashboard**: Static HTML + JavaScript (no framework), Leaflet/Mapbox for map view
- **LLM**: Self-hosted Llama/Mistral-class model (for production narrative generation)
- **Deployment**: Docker (edge devices), NVIDIA Jetson or equivalent (edge compute)

## Project Structure

```
ibvap_demo/
├── camera_manager.py          # Entry point: spawns per-camera workers + matcher
├── pipeline.py                # Per-camera AI pipeline (Steps 6-13, 18)
├── reid_matcher.py            # Cross-camera identity matching (Steps 19-21)
├── face_engine.py             # InsightFace wrapper with best-shot persistence
├── reid_embedder.py           # OSNet embedding generation
├── fence_setup.py             # Interactive fence line definition tool
├── camera_registry.json       # Camera metadata: ID, location, source, zone
├── config.json                # Fence coordinates, Re-ID params, zone policies
├── custom_tracker.yaml        # ByteTrack configuration
├── requirements.txt           # Python dependencies
├── models/                    # Model weights
│   ├── insightface/buffalo_l/ # SCRFD + ArcFace ONNX models
│   └── osnet_x1_0_msmt17.pt   # OSNet Re-ID checkpoint
├── known_faces/               # Enrollment images (optional)
├── web/                       # Dashboard and output files
│   ├── index.html             # Dashboard (static)
│   ├── cameras.json           # Camera registry with fence info
│   ├── reid_events.jsonl      # Append-only Re-ID embedding log
│   ├── reid_mapping.json      # Local track → Global ID mapping
│   ├── global_identities.json # Global identity table
│   ├── disambiguation_queue.json # Ambiguous Re-ID matches
│   ├── alerts_camX.json       # Per-camera fence crossing alerts
│   ├── detections_camX.json   # Per-camera detection log
│   ├── faces_camX.json        # Per-camera face recognition log
│   ├── health_camX.json       # Per-camera health status
│   ├── latest_frame_camX.jpg  # Live feed snapshot per camera
│   └── snapshots/             # Evidence clips
├── video.mp4                  # Demo footage: cam1 (person crossing)
├── video2.mp4                 # Demo footage: cam2 (person crossing)
├── video3.mp4                 # Demo footage: cam3 (person crossing)
└── video4.mp4                 # Demo footage: cam4 (parked vehicle)
```

## Setup and Run Instructions

### Prerequisites

- Python 3.10 or newer
- NVIDIA GPU with CUDA support (tested on RTX 5060 Laptop GPU, sm_120)
- CUDA Toolkit 12.4+
- cuDNN 9.x

### Installation

1. Clone the repository and navigate to the project directory:

```bash
cd ibvap_demo
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Verify GPU setup:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

If CUDA is not available or you see kernel compatibility errors with RTX 5060, reinstall PyTorch with cu128 support:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

5. Uninstall conflicting onnxruntime packages:

InsightFace depends on the CPU `onnxruntime` wheel, but we need `onnxruntime-gpu` for CUDA acceleration. Uninstall the CPU version after InsightFace installation:

```bash
pip uninstall -y onnxruntime
```

Verify `onnxruntime-gpu==1.26.0` is installed and CUDAExecutionProvider is available:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Expected output should include `'CUDAExecutionProvider'`.

### Optional: Add Enrollment Photos

Place enrollment images in `known_faces/` directory. Filename stem becomes the display name:

```bash
mkdir known_faces
# Copy enrollment photos, e.g., Krish.jpg, Officer_Sharma.png
```

If the directory is empty, face detection still runs but all faces are labeled "unknown" or "no_usable_face."

### Optional: Define Virtual Fence Lines

Run the interactive fence setup tool for each camera:

```bash
python fence_setup.py cam1
```

This opens a window showing a frame from cam1's video. Click two points to define the border line, then press any key to save. Repeat for cam2, cam3, cam4 as needed. Fence coordinates are saved to `config.json`.

### Run the Demo

Start all camera workers, the Re-ID matcher, and the dashboard file merger:

```bash
python camera_manager.py --device cuda --loop
```

Flags:

- `--device cuda`: Require GPU and error out if unavailable (recommended to catch silent CPU fallback)
- `--device cpu`: Force CPU (for testing without GPU)
- `--device auto`: Use GPU if available, fall back to CPU with a warning (default)
- `--loop`: Restart each camera's clip when it ends, for continuous demo
- `--model yolov8s.pt`: Model weights (default: `yolov8s.pt`, can use `yolov8n.pt` for faster but less accurate detection)
- `--imgsz 1280`: Input resolution (default: 1280, can reduce to 960 or 640 for lower GPU memory usage)

In a separate terminal, serve the dashboard:

```bash
cd web
python -m http.server
```

Open `http://localhost:8000` in a browser. The dashboard polls JSON files every second and updates live.

### Expected Output

Console output shows:

```
Prefetching face (buffalo_l) and Re-ID (OSNet) weights if needed...
OSNet x1.0 Re-ID embedder loaded on cuda:0 (41 tensors, 512-d).
InsightFace buffalo_l face engine loaded on cuda:0.
[cam1] opened video.mp4: ~26.0s at 25.0 fps (725 frames).
[cam2] opened video2.mp4: ~26.0s at 25.0 fps (725 frames).
[cam3] opened video3.mp4: ~26.0s at 25.0 fps (725 frames).
[cam4] opened video4.mp4: ~26.0s at 25.0 fps (725 frames).
IBVAP multi-camera demo running — 4 camera(s). Open web/index.html (serve the web/ folder).
Looping enabled (--loop) — cameras restart their clip on end. Press Ctrl+C to stop.
```

Dashboard shows:

- Four per-camera panels with live feeds updating
- Fence crossing alerts (if fences are configured and person crosses)
- Face detection labels: "no_usable_face" for the hooded subject in demo footage
- Cross-camera section showing global identity G0001 moving cam1 → cam2 → cam3

### Demo Footage Notes

The provided clips are intentionally challenging:

- `video.mp4`, `video2.mp4`, `video3.mp4`, `video4.mp4`: Same person crossing cam1 → cam2 → cam3. Low-res night/IR footage, subject wears a hoodie with the hood up, crouched posture, face not visible. Expected outcome: face status shows "no_usable_face" (correct, not a failure), global Re-ID correctly links as one identity across all three cameras.
- `video4.mp4`: Parked vehicle at night, different location.

A positive face recognition match requires adding enrollment photos to `known_faces/` and footage where a face is clearly visible. The demo clips are designed to test the "no face visible" path, which is common in real border surveillance scenarios.

## Limitations 

- Camera adjacency graph uses a simplified time-window model. Production deployment requires surveyed, terrain-grounded adjacency edges.
- Thermal camera support (Step 50) requires fine-tuning on elevated, static thermal footage after initial FLIR ADAS tuning.
- Posture and micro-behavior signals (Step 12) are partially implemented. MediaPipe Pose integration and bounding-box width/height ratio tracking are placeholders.
- Group spacing analysis (Step 17) discrete group entity tracking is not implemented. Behavioral spacing-variance signal is a placeholder.

### Plan

1. Complete policy engine, context reasoning, full risk scoring
2. Implement blockchain evidence layer with Hyperledger Fabric
3. Build command dashboard backend (REST/WebSocket API)
4. Integrate with real RTSP camera streams
5. Deploy edge compute units for bandwidth-constrained sites
6. Add generative AI narrative generation with self-hosted LLM
7. Implement agentic investigation assistant
8. Add IPFS private swarm for evidence storage
9. Apply Phase 11 cost optimizations (motion-gating, quantization, cascade routing)
10. Pilot deployment at a single BOP with real surveyed adjacency graph
11. Scale to multi-BOP deployment with central coordination

-
