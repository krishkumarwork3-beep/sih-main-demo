"""OSNet x1.0 person Re-ID embeddings (torch, CUDA).

Body/appearance feature used for cross-camera matching — deliberately not
the ArcFace face embedding, so a hooded/turned-away subject still matches.
"""
import os

import cv2
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "osnet_x1_0_msmt17.pt")
WEIGHTS_DRIVE_ID = "1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x"
INPUT_H, INPUT_W = 256, 128
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_EMBEDDER = None
_EMBEDDER_FAILED = False


def prefetch_reid_weights():
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    if os.path.isfile(WEIGHTS_PATH) and os.path.getsize(WEIGHTS_PATH) > 1_000_000:
        return WEIGHTS_PATH
    import gdown
    url = f"https://drive.google.com/uc?id={WEIGHTS_DRIVE_ID}"
    print(f"Downloading OSNet x1.0 MSMT17 weights to {WEIGHTS_PATH} ...")
    gdown.download(url, WEIGHTS_PATH, quiet=False)
    if not os.path.isfile(WEIGHTS_PATH):
        raise RuntimeError("OSNet weight download failed.")
    return WEIGHTS_PATH


def _load_state_dict(model, weight_path):
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    matched = 0
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state[k] = v
            matched += 1
    if matched == 0:
        raise RuntimeError(f"OSNet weights in {weight_path} did not match osnet_x1_0.")
    model_dict.update(new_state)
    model.load_state_dict(model_dict)
    return matched


class ReidEmbedder:
    def __init__(self, device="cuda"):
        from torchreid.reid.models.osnet import osnet_x1_0

        if device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("OSNet requested CUDA but torch.cuda.is_available() is False.")
        self.device = torch.device("cuda:0" if device != "cpu" else "cpu")
        prefetch_reid_weights()
        model = osnet_x1_0(num_classes=1, pretrained=False, loss="softmax")
        matched = _load_state_dict(model, WEIGHTS_PATH)
        model.eval()
        model.to(self.device)
        self.model = model
        # Force a real kernel so a silent CPU/incompatible-GPU failure surfaces now.
        dummy = torch.zeros(1, 3, INPUT_H, INPUT_W, device=self.device)
        with torch.no_grad():
            out = model(dummy)
        dim = int(out.reshape(1, -1).shape[1])
        print(f"OSNet x1.0 Re-ID embedder loaded on {self.device} "
              f"({matched} tensors, {dim}-d).")

    def embed(self, bgr_crop):
        if bgr_crop is None or bgr_crop.size == 0:
            return None
        if bgr_crop.shape[0] < 16 or bgr_crop.shape[1] < 8:
            return None
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        x = img.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = np.transpose(x, (2, 0, 1))[None, ...]
        tensor = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
        vec = feat.detach().float().cpu().numpy().reshape(-1)
        n = float(np.linalg.norm(vec))
        if n < 1e-8:
            return None
        return (vec / n).astype(np.float32)


def get_reid_embedder(device="cuda"):
    global _EMBEDDER, _EMBEDDER_FAILED
    if _EMBEDDER is not None:
        return _EMBEDDER
    if _EMBEDDER_FAILED:
        return None
    try:
        _EMBEDDER = ReidEmbedder(device=device)
        return _EMBEDDER
    except Exception as exc:
        _EMBEDDER_FAILED = True
        print(f"OSNet Re-ID embedder unavailable ({exc}) — cross-camera appearance "
              f"matching will be skipped.")
        return None


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
