"""SCRFD detection + ArcFace recognition via InsightFace / onnxruntime.

Replaces the Haar-cascade presence check. Gallery images in known_faces/
(filename stem = display name) are embedded once at load time. An empty
gallery is valid: every usable face is labeled ``unknown``.
"""
import os
import glob

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
INSIGHTFACE_ROOT = os.path.join(BASE_DIR, "models", "insightface")
BUFFALO = "buffalo_l"

# Quality / matching knobs. Adaptive cosine cutoff: a frontal, large, high-
# confidence crop can match at a lower similarity; a small/angled crop must
# clear a higher bar so we don't stamp a stranger as a name (or vice versa).
MIN_FACE_SIDE_PX = 16
MIN_QUALITY_TO_EMBED = 0.04
MIN_QUALITY_TO_CONFIRM_UNKNOWN = 0.22
SIM_HIGH_QUALITY = 0.32
SIM_LOW_QUALITY = 0.50

_ENGINE = None
_ENGINE_FAILED = False


def prefetch_face_models():
    """Download buffalo_l (SCRFD + ArcFace) without constructing GPU sessions."""
    from insightface.utils import ensure_available
    os.makedirs(INSIGHTFACE_ROOT, exist_ok=True)
    ensure_available("models", BUFFALO, root=INSIGHTFACE_ROOT)


def _l2_normalize(vec):
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(vec))
    if n < 1e-8:
        return vec
    return vec / n


def _frontalness(kps):
    """1.0 = eyes/mouth symmetric about the nose (frontal); ~0 = strong profile."""
    if kps is None or len(kps) < 5:
        return 0.5
    le, re, nose, lm, rm = np.asarray(kps, dtype=np.float32)[:5]
    eye_span = abs(float(re[0] - le[0])) + 1e-6
    mouth_span = abs(float(rm[0] - lm[0])) + 1e-6
    eye_mid = 0.5 * (le[0] + re[0])
    mouth_mid = 0.5 * (lm[0] + rm[0])
    eye_off = abs(float(nose[0] - eye_mid)) / eye_span
    mouth_off = abs(float(nose[0] - mouth_mid)) / mouth_span
    return float(max(0.0, 1.0 - 2.0 * eye_off) * max(0.0, 1.0 - 2.0 * mouth_off))


def face_quality(det_score, bbox, kps):
    x1, y1, x2, y2 = bbox
    w = max(0.0, float(x2 - x1))
    h = max(0.0, float(y2 - y1))
    if w < MIN_FACE_SIDE_PX or h < MIN_FACE_SIDE_PX:
        return 0.0
    size_norm = min(((w * h) ** 0.5) / 80.0, 1.0)
    front = _frontalness(kps)
    return float(max(0.0, det_score)) * size_norm * (0.35 + 0.65 * front)


def adaptive_sim_threshold(quality):
    q = max(0.0, min(1.0, float(quality)))
    return SIM_LOW_QUALITY - (SIM_LOW_QUALITY - SIM_HIGH_QUALITY) * q


def _providers(want_cuda):
    import onnxruntime as ort
    available = ort.get_available_providers()
    if want_cuda:
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is not available in onnxruntime — "
                "uninstall the CPU `onnxruntime` package and keep onnxruntime-gpu. "
                "InsightFace cannot run on GPU."
            )
        cuda_opts = {
            "device_id": "0",
            "arena_extend_strategy": "kSameAsRequested",
            "gpu_mem_limit": str(384 * 1024 * 1024),
            "cudnn_conv_algo_search": "DEFAULT",
            "cudnn_conv_use_max_workspace": "0",
            "do_copy_in_default_stream": "1",
        }
        return [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _session_provider(session):
    try:
        return session.get_providers()[0]
    except Exception:
        return "unknown"


class TrackFaceState:
    """Best-shot-per-track + persistent identity label."""

    def __init__(self):
        self.best_quality = 0.0
        self.best_embedding = None
        self.label = "no_usable_face"
        self.confirmed = False
        self.last_bbox = None  # last observed face box (full-frame xyxy)

    def update_from_detection(self, det, gallery):
        """Incorporate one face detection. ``det`` has bbox, kps, det_score, embedding."""
        bbox = det["bbox"]
        self.last_bbox = bbox
        quality = face_quality(det["det_score"], bbox, det.get("kps"))
        if quality < MIN_QUALITY_TO_EMBED:
            return

        embedding = det.get("embedding")
        if embedding is None:
            return
        embedding = _l2_normalize(embedding)

        if quality + 1e-6 < self.best_quality:
            return

        self.best_quality = quality
        self.best_embedding = embedding

        name, sim = match_gallery(embedding, gallery, quality)
        if name is not None:
            self.label = name
            self.confirmed = True
            return

        if self.confirmed and self.label not in ("unknown", "no_usable_face"):
            # Already have a name — never flip it back on a worse/no match.
            return

        if quality >= MIN_QUALITY_TO_CONFIRM_UNKNOWN:
            self.label = "unknown"
            self.confirmed = True
        elif self.label == "no_usable_face":
            # Saw a face but not clean enough to confirm stranger vs enrolled.
            pass


def match_gallery(embedding, gallery, quality):
    if not gallery:
        return None, None
    thresh = adaptive_sim_threshold(quality)
    best_name, best_sim = None, -1.0
    for name, gvec in gallery:
        sim = float(np.dot(embedding, gvec))
        if sim > best_sim:
            best_sim, best_name = sim, name
    if best_name is not None and best_sim >= thresh:
        return best_name, best_sim
    return None, best_sim


class FaceEngine:
    def __init__(self, want_cuda=True):
        self.app = None
        self.gallery = []  # list of (name, embedding)
        self.provider = None
        from insightface.app import FaceAnalysis

        os.makedirs(INSIGHTFACE_ROOT, exist_ok=True)
        providers = _providers(want_cuda)
        self.app = FaceAnalysis(
            name=BUFFALO,
            root=INSIGHTFACE_ROOT,
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        ctx_id = 0 if want_cuda else -1
        self.app.prepare(ctx_id=ctx_id, det_size=(320, 320), det_thresh=0.45)

        det_prov = _session_provider(getattr(self.app.det_model, "session", None))
        rec = self.app.models.get("recognition")
        rec_prov = _session_provider(getattr(rec, "session", None)) if rec is not None else det_prov
        self.provider = det_prov
        if want_cuda and (not str(det_prov).startswith("CUDA") or not str(rec_prov).startswith("CUDA")):
            raise RuntimeError(
                f"InsightFace loaded but not on CUDA (det={det_prov}, rec={rec_prov})."
            )
        print(f"InsightFace buffalo_l (SCRFD+ArcFace) loaded on {det_prov} "
              f"(recognition: {rec_prov}).")
        self.gallery = load_gallery(self)

    def detect(self, bgr_frame):
        if self.app is None:
            return []
        try:
            faces = self.app.get(bgr_frame)
        except Exception:
            return []
        out = []
        for face in faces:
            bbox = np.asarray(face.bbox, dtype=np.float32)
            kps = None if face.kps is None else np.asarray(face.kps, dtype=np.float32)
            emb = None if getattr(face, "embedding", None) is None else np.asarray(face.embedding, dtype=np.float32)
            out.append({
                "bbox": bbox,
                "kps": kps,
                "det_score": float(face.det_score),
                "embedding": emb,
            })
        return out

    def embed_image(self, bgr_img):
        dets = self.detect(bgr_img)
        if not dets:
            return None
        best = max(dets, key=lambda d: face_quality(d["det_score"], d["bbox"], d.get("kps")))
        if best.get("embedding") is None:
            return None
        return _l2_normalize(best["embedding"])


def load_gallery(engine):
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    gallery = []
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        paths.extend(glob.glob(os.path.join(KNOWN_FACES_DIR, ext)))
        paths.extend(glob.glob(os.path.join(KNOWN_FACES_DIR, ext.upper())))
    seen = set()
    for path in sorted(paths):
        if path in seen:
            continue
        seen.add(path)
        name = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path)
        if img is None:
            print(f"known_faces: could not read {path} — skipped.")
            continue
        emb = engine.embed_image(img)
        if emb is None:
            print(f"known_faces: no face in {path} — skipped.")
            continue
        gallery.append((name, emb))
        print(f"known_faces: enrolled '{name}' from {os.path.basename(path)}.")
    if not gallery:
        print("known_faces: gallery empty — faces will be labeled unknown / "
              "no_usable_face (pipeline still runs).")
    return gallery


def get_face_engine(want_cuda=True):
    """Process-local singleton. Returns None if load fails (caller degrades)."""
    global _ENGINE, _ENGINE_FAILED
    if _ENGINE is not None:
        return _ENGINE
    if _ENGINE_FAILED:
        return None
    try:
        _ENGINE = FaceEngine(want_cuda=want_cuda)
        return _ENGINE
    except Exception as exc:
        _ENGINE_FAILED = True
        print(f"InsightFace unavailable ({exc}) — face detection/recognition will be skipped.")
        return None


def associate_faces_to_box(faces, person_box):
    """Return the highest-quality face whose center lies inside the person box."""
    x1, y1, x2, y2 = person_box
    best, best_q = None, -1.0
    for det in faces:
        bx1, by1, bx2, by2 = det["bbox"]
        cx, cy = 0.5 * (bx1 + bx2), 0.5 * (by1 + by2)
        if not (x1 <= cx <= x2 and y1 <= cy <= y2):
            continue
        q = face_quality(det["det_score"], det["bbox"], det.get("kps"))
        if q > best_q:
            best, best_q = det, q
    return best
