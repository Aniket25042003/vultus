import base64
import io
import os
from typing import Any, Optional, Tuple, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

try:
    import numpy as np
except ImportError as exc:
    np = None

try:
    import torch
    from torch import nn
except ImportError as exc:
    torch = None
    nn = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    from ultralytics import YOLO as _YOLO
except ImportError:
    _YOLO = None

app = FastAPI(title="Vultus ML", version="0.1.0")

MODEL_PATH = os.environ.get(
    "FACE_MODEL_PATH", "/Users/aniketpatel/Desktop/FaceDetection/backbone_best.pt"
)
FACE_DETECT_MODEL_PATH = os.environ.get(
    "FACE_DETECT_MODEL_PATH", "/Users/aniketpatel/Desktop/FaceDetection/yolov8l_100e.pt"
)
EMOTION_MODEL_PATH = os.environ.get(
    "EMOTION_MODEL_PATH", "/Users/aniketpatel/Desktop/FaceDetection/emotion-ferplus-8.onnx"
)
INPUT_SIZE = int(os.environ.get("FACE_INPUT_SIZE", "112"))
FACE_DETECT_INPUT_SIZE = int(os.environ.get("FACE_DETECT_INPUT_SIZE", "640"))
EMOTION_INPUT_SIZE = int(os.environ.get("EMOTION_INPUT_SIZE", "64"))
FACE_DETECT_CONFIDENCE = float(os.environ.get("FACE_DETECT_CONFIDENCE", "0.35"))


class FaceRequest(BaseModel):
    frame_id: Optional[str] = None
    image_base64: Optional[str] = None
    source: Optional[str] = None
    face_bbox: Optional[Tuple[float, float, float, float]] = None


class EmbeddingRequest(BaseModel):
    frame_id: Optional[str] = None
    image_base64: Optional[str] = None
    source: Optional[str] = None
    face_bbox: Optional[Tuple[float, float, float, float]] = None


class EmotionRequest(BaseModel):
    frame_id: Optional[str] = None
    image_base64: Optional[str] = None
    source: Optional[str] = None
    face_bbox: Optional[Tuple[float, float, float, float]] = None


class PersonRequest(BaseModel):
    frame_id: Optional[str] = None
    image_base64: Optional[str] = None
    source: Optional[str] = None


# ---------------------------------------------------------------------------
# IResNet architecture (from train.py) – needed to load state-dict checkpoints
# ---------------------------------------------------------------------------

class _SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.avg_pool(x)
        w = self.fc1(w)
        w = self.relu(w)
        w = self.fc2(w)
        w = self.sigmoid(w)
        return x * w


class _IRSEBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_se=True, bn_momentum=0.9, bn_eps=1e-5):
        super().__init__()
        self.use_shortcut = (in_ch == out_ch and stride == 1)
        self.bn0 = nn.BatchNorm2d(in_ch, eps=bn_eps, momentum=bn_momentum)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch, eps=bn_eps, momentum=bn_momentum)
        self.prelu = nn.PReLU(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch, eps=bn_eps, momentum=bn_momentum)
        self.se = _SEModule(out_ch) if use_se else nn.Identity()
        if self.use_shortcut:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_ch, eps=bn_eps, momentum=bn_momentum),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        return out + identity


class _IResNet(nn.Module):
    def __init__(self, layers, embedding_dim=512, dropout=0.4, use_se=True,
                 bn_momentum=0.9, bn_eps=1e-5):
        super().__init__()
        self.in_ch = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=bn_eps, momentum=bn_momentum)
        self.prelu = nn.PReLU(64)
        self.stage1 = self._make_layer(64,  layers[0], stride=2, use_se=use_se,
                                       bn_momentum=bn_momentum, bn_eps=bn_eps)
        self.stage2 = self._make_layer(128, layers[1], stride=2, use_se=use_se,
                                       bn_momentum=bn_momentum, bn_eps=bn_eps)
        self.stage3 = self._make_layer(256, layers[2], stride=2, use_se=use_se,
                                       bn_momentum=bn_momentum, bn_eps=bn_eps)
        self.stage4 = self._make_layer(512, layers[3], stride=2, use_se=use_se,
                                       bn_momentum=bn_momentum, bn_eps=bn_eps)
        self.bn2 = nn.BatchNorm2d(512, eps=bn_eps, momentum=bn_momentum)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * 7 * 7, embedding_dim, bias=True)
        self.emb_bn = nn.BatchNorm1d(embedding_dim, eps=bn_eps, momentum=bn_momentum)

    def _make_layer(self, out_ch, blocks, stride, use_se, bn_momentum, bn_eps):
        layers = [_IRSEBlock(self.in_ch, out_ch, stride=stride, use_se=use_se,
                             bn_momentum=bn_momentum, bn_eps=bn_eps)]
        self.in_ch = out_ch
        for _ in range(1, blocks):
            layers.append(_IRSEBlock(self.in_ch, out_ch, stride=1, use_se=use_se,
                                     bn_momentum=bn_momentum, bn_eps=bn_eps))
        return nn.Sequential(*layers)

    def forward(self, x):
        import torch.nn.functional as _F
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.emb_bn(x)
        x = _F.normalize(x, p=2, dim=1)
        return x


_KNOWN_ARCHITECTURES = {
    "iresnet100": [2, 9, 20, 2],
    "iresnet200": [6, 26, 60, 6],
}


def _build_iresnet(layers, embedding_dim=512, dropout=0.4):
    """Construct an IResNet backbone given layer counts."""
    return _IResNet(layers, embedding_dim=embedding_dim, dropout=dropout)


def _guess_architecture(state_dict) -> Optional[list]:
    """Inspect the state dict keys to figure out the layer counts."""
    max_idx = {1: 0, 2: 0, 3: 0, 4: 0}
    for key in state_dict.keys():
        for stage in range(1, 5):
            prefix = f"stage{stage}."
            if key.startswith(prefix):
                parts = key[len(prefix):].split(".")
                try:
                    idx = int(parts[0])
                    if idx + 1 > max_idx[stage]:
                        max_idx[stage] = idx + 1
                except (ValueError, IndexError):
                    pass
    layers = [max_idx[1], max_idx[2], max_idx[3], max_idx[4]]
    if all(v > 0 for v in layers):
        return layers
    return None


# ---------------------------------------------------------------------------


class ModelState:
    def __init__(self) -> None:
        self.model = None
        self.error: Optional[str] = None
        self.model_type: str = "unloaded"

    def load(self) -> None:
        print(f"[VULTUS-ML] Attempting to load model from: {MODEL_PATH}")
        if torch is None:
            self.error = "torch not installed"
            print(f"[VULTUS-ML] Error: {self.error}")
            return
        if not MODEL_PATH or not os.path.exists(MODEL_PATH):
            self.error = f"model not found at {MODEL_PATH}"
            print(f"[VULTUS-ML] Error: {self.error}")
            return

        # 1. Try TorchScript
        try:
            print("[VULTUS-ML] Loading TorchScript model...")
            self.model = torch.jit.load(MODEL_PATH, map_location="cpu")
            self.model.eval()
            self.model_type = "torchscript"
            self.error = None
            print("[VULTUS-ML] Successfully loaded TorchScript model.")
            return
        except Exception as e:
            print(f"[VULTUS-ML] TorchScript load failed: {e}")

        # 2. Try direct nn.Module
        try:
            print("[VULTUS-ML] Loading as PyTorch module...")
            loaded = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            if isinstance(loaded, nn.Module):
                loaded.eval()
                self.model = loaded
                self.model_type = "pytorch-module"
                self.error = None
                print("[VULTUS-ML] Successfully loaded PyTorch module.")
                return
        except Exception as exc:
            print(f"[VULTUS-ML] Direct module load failed: {exc}")
            loaded = None

        # 3. Load as state-dict checkpoint (the format used by train.py)
        try:
            if loaded is None:
                loaded = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

            # Extract the state dict and metadata
            if isinstance(loaded, dict) and "backbone" in loaded:
                state_dict = loaded["backbone"]
                embedding_dim = loaded.get("embedding_dim", 512)
                print(f"[VULTUS-ML] Detected train.py checkpoint (embedding_dim={embedding_dim})")
            elif isinstance(loaded, dict):
                state_dict = loaded
                embedding_dim = 512
                print("[VULTUS-ML] Detected raw state-dict")
            else:
                self.error = f"Unexpected checkpoint type: {type(loaded)}"
                print(f"[VULTUS-ML] Error: {self.error}")
                return

            # Determine architecture from state dict keys
            layers = _guess_architecture(state_dict)
            if layers is None:
                # Fallback to iresnet100 (the default in train.py)
                layers = _KNOWN_ARCHITECTURES["iresnet100"]
                print(f"[VULTUS-ML] Could not guess architecture, defaulting to iresnet100 {layers}")
            else:
                arch_name = "unknown"
                for name, cfg in _KNOWN_ARCHITECTURES.items():
                    if cfg == layers:
                        arch_name = name
                        break
                print(f"[VULTUS-ML] Detected architecture: {arch_name} (layers={layers})")

            model = _build_iresnet(layers, embedding_dim=embedding_dim, dropout=0.0)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            self.model = model
            self.model_type = "iresnet-statedict"
            self.error = None
            print(f"[VULTUS-ML] Successfully loaded IResNet from state-dict "
                  f"(layers={layers}, dim={embedding_dim})")
        except Exception as exc:
            self.error = str(exc)
            print(f"[VULTUS-ML] State-dict load failed: {self.error}")


STATE = ModelState()
STATE.load()


# ---------------------------------------------------------------------------
# YOLOv8-Face detector (replaces ONNX person detector)
# Uses ultralytics YOLO API: https://github.com/Yusepp/YOLOv8-Face
# ---------------------------------------------------------------------------

class FaceDetectorState:
    def __init__(self, path: str) -> None:
        self.path = path
        self.model = None
        self.error: Optional[str] = None

    def load(self) -> None:
        print(f"[VULTUS-ML] Attempting to load YOLOv8-Face model from: {self.path}")
        if _YOLO is None:
            self.error = "ultralytics not installed"
            print(f"[VULTUS-ML] Error: {self.error}")
            return
        if not self.path or not os.path.exists(self.path):
            self.error = f"model not found at {self.path}"
            print(f"[VULTUS-ML] Error: {self.error}")
            return
        try:
            model = _YOLO(self.path)
            # Use CPU inside Docker (no GPU); locally use MPS/CUDA if available
            if torch is not None and torch.cuda.is_available():
                model.to("cuda")
            elif torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                model.to("mps")
            else:
                model.to("cpu")
            self.model = model
            self.error = None
            print(f"[VULTUS-ML] Loaded YOLOv8-Face model (names={model.names}).")
        except Exception as exc:
            self.error = str(exc)
            print(f"[VULTUS-ML] Error loading YOLOv8-Face: {self.error}")


# ONNX state is still used for the emotion model
class OnnxState:
    def __init__(self, path: str) -> None:
        self.path = path
        self.session = None
        self.error: Optional[str] = None
        self.input_name: Optional[str] = None
        self.input_shape: Optional[Tuple[Any, ...]] = None

    def load(self) -> None:
        print(f"[VULTUS-ML] Attempting to load ONNX model from: {self.path}")
        if ort is None:
            self.error = "onnxruntime not installed"
            print(f"[VULTUS-ML] Error: {self.error}")
            return
        if not self.path or not os.path.exists(self.path):
            self.error = f"model not found at {self.path}"
            print(f"[VULTUS-ML] Error: {self.error}")
            return
        try:
            session = ort.InferenceSession(self.path, providers=["CPUExecutionProvider"])
            inputs = session.get_inputs()
            self.session = session
            self.input_name = inputs[0].name if inputs else None
            self.input_shape = tuple(inputs[0].shape) if inputs else None
            self.error = None
            print(f"[VULTUS-ML] Loaded ONNX model with input {self.input_name}.")
        except Exception as exc:
            self.error = str(exc)
            print(f"[VULTUS-ML] Error: {self.error}")


FACE_DETECT_STATE = FaceDetectorState(FACE_DETECT_MODEL_PATH)
EMOTION_STATE = OnnxState(EMOTION_MODEL_PATH)
FACE_DETECT_STATE.load()
EMOTION_STATE.load()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": STATE.model is not None,
        "model_type": STATE.model_type,
        "face_detect_model_loaded": FACE_DETECT_STATE.model is not None,
        "emotion_model_loaded": EMOTION_STATE.session is not None,
        "error": STATE.error,
        "face_detect_error": FACE_DETECT_STATE.error,
        "emotion_error": EMOTION_STATE.error,
    }


def _decode_image(image_base64: str) -> Image.Image:
    raw = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _preprocess(image: Image.Image) -> "torch.Tensor":
    image = image.resize((INPUT_SIZE, INPUT_SIZE))
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor


def _to_numpy(image: Image.Image, size: int, channels: int) -> "np.ndarray":
    if np is None:
        raise RuntimeError("numpy not installed")
    if channels == 1:
        image = image.convert("L")
    else:
        image = image.convert("RGB")
    image = image.resize((size, size))
    array = np.asarray(image, dtype=np.float32)
    if channels == 1:
        array = array[:, :, None]
    array = array / 255.0
    array = np.transpose(array, (2, 0, 1))
    return np.expand_dims(array, axis=0)


def _detect_face(image: Image.Image) -> Optional[Tuple[float, float, float, float, float]]:
    """Run YOLOv8-Face detection and return the best face bounding box.

    Uses the ultralytics YOLO API (https://github.com/Yusepp/YOLOv8-Face).
    Returns (x1, y1, x2, y2, confidence) in original image coordinates,
    or None if no face is detected.
    """
    if FACE_DETECT_STATE.model is None or np is None:
        return None
    # Convert PIL to numpy array (RGB, H×W×C) for ultralytics
    img_array = np.asarray(image.convert("RGB"))
    # Run inference — ultralytics handles letterbox, NMS, etc. internally
    results = FACE_DETECT_STATE.model.predict(
        img_array,
        imgsz=FACE_DETECT_INPUT_SIZE,
        conf=FACE_DETECT_CONFIDENCE,
        verbose=False,
    )
    if not results or len(results) == 0:
        return None
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    # Pick the detection with the highest confidence
    confs = boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confs))
    score = float(confs[best_idx])
    xyxy = boxes.xyxy.cpu().numpy()[best_idx]  # [x1, y1, x2, y2]
    x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2, score


def _detect_all_faces(image: Image.Image) -> List[Tuple[float, float, float, float, float]]:
    """Run YOLOv8-Face and return ALL detected face bounding boxes."""
    if FACE_DETECT_STATE.model is None or np is None:
        return []
    img_array = np.asarray(image.convert("RGB"))
    results = FACE_DETECT_STATE.model.predict(
        img_array,
        imgsz=FACE_DETECT_INPUT_SIZE,
        conf=FACE_DETECT_CONFIDENCE,
        verbose=False,
    )
    if not results or len(results) == 0:
        return []
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    confs = boxes.conf.cpu().numpy()
    xyxys = boxes.xyxy.cpu().numpy()
    detections = []
    for i in range(len(confs)):
        x1, y1, x2, y2 = float(xyxys[i][0]), float(xyxys[i][1]), float(xyxys[i][2]), float(xyxys[i][3])
        if x2 > x1 and y2 > y1:
            detections.append((x1, y1, x2, y2, float(confs[i])))
    # Sort by confidence descending
    detections.sort(key=lambda d: d[4], reverse=True)
    return detections


def _face_crop(image: Image.Image, bbox: Optional[Tuple[float, float, float, float]], padding: float = 0.25) -> Image.Image:
    """Crop face from image using the YOLOv8-Face bounding box.

    Since YOLOv8-Face returns the face bounding box directly, we just add
    symmetric padding around it and make the crop square for the embedding model.
    """
    if bbox is None:
        min_dim = min(image.size)
        left = (image.size[0] - min_dim) // 2
        top = (image.size[1] - min_dim) // 2
        return image.crop((left, top, left + min_dim, top + min_dim))
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    # Add padding around the face bbox
    pad_x = w * padding
    pad_y = h * padding
    crop_x1 = x1 - pad_x
    crop_y1 = y1 - pad_y
    crop_x2 = x2 + pad_x
    crop_y2 = y2 + pad_y
    # Make the crop square (IResNet expects square input)
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    if crop_w > crop_h:
        diff = crop_w - crop_h
        crop_y1 -= diff / 2
        crop_y2 += diff / 2
    elif crop_h > crop_w:
        diff = crop_h - crop_w
        crop_x1 -= diff / 2
        crop_x2 += diff / 2
    # Clip to image bounds
    iw, ih = image.size
    crop_x1 = max(0.0, crop_x1)
    crop_y1 = max(0.0, crop_y1)
    crop_x2 = min(float(iw), crop_x2)
    crop_y2 = min(float(ih), crop_y2)
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return image
    return image.crop((crop_x1, crop_y1, crop_x2, crop_y2))


def _encode_image_base64(image: Image.Image, quality: int = 85) -> str:
    """Encode a PIL image to JPEG base64 string."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _normalize_embedding(vector: "np.ndarray") -> "np.ndarray":
    if np is None:
        return vector
    norm = np.linalg.norm(vector) + 1e-9
    return vector / norm


def _infer_embedding_vector(image: Image.Image) -> Optional["np.ndarray"]:
    if torch is None or STATE.model is None or np is None:
        return None
    tensor = _preprocess(image)
    with torch.no_grad():
        output = STATE.model(tensor)
    if hasattr(output, "detach"):
        vector = output.detach().cpu().flatten().numpy().astype("float32")
        return _normalize_embedding(vector)
    return None


EMOTION_LABELS = [
    "angry",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


def _infer_emotion(image: Image.Image) -> Optional[dict]:
    if EMOTION_STATE.session is None or np is None:
        return None
    tensor = _to_numpy(image, EMOTION_INPUT_SIZE, 1)
    outputs = EMOTION_STATE.session.run(None, {EMOTION_STATE.input_name: tensor})
    if not outputs:
        return None
    scores = outputs[0]
    scores = np.squeeze(scores)
    if scores.ndim != 1:
        scores = scores.flatten()
    exp = np.exp(scores - np.max(scores))
    probs = exp / (np.sum(exp) + 1e-9)
    idx = int(np.argmax(probs))
    return {
        "emotion": EMOTION_LABELS[idx] if idx < len(EMOTION_LABELS) else "unknown",
        "confidence": float(probs[idx]),
        "scores": {EMOTION_LABELS[i]: float(probs[i]) for i in range(min(len(EMOTION_LABELS), probs.size))},
    }


@app.post("/predict/person")
async def predict_person(payload: PersonRequest):
    """Detect faces in a frame using YOLOv8-Face.

    This endpoint is kept as /predict/person for API compatibility but now
    detects faces directly using the YOLOv8-Face model.
    """
    print(f"[VULTUS-ML] Face detection request received for frame: {payload.frame_id or 'unknown'}")
    if not payload.image_base64:
        raise HTTPException(status_code=400, detail="image_base64 required")
    try:
        image = _decode_image(payload.image_base64)
    except Exception as exc:
        return {
            "detected": False,
            "confidence": 0.0,
            "model": "yolov8-face-error",
            "error": str(exc),
        }
    detection = _detect_face(image)
    if detection is None:
        return {
            "detected": False,
            "confidence": 0.0,
            "model": "yolov8-face",
        }
    x1, y1, x2, y2, score = detection
    return {
        "detected": True,
        "confidence": round(score, 4),
        "bbox": [x1, y1, x2, y2],
        "model": "yolov8-face",
    }


@app.post("/predict/embedding")
async def predict_embedding(payload: EmbeddingRequest):
    print(f"[VULTUS-ML] Embedding request received for frame: {payload.frame_id or 'unknown'}")
    if not payload.image_base64:
        raise HTTPException(status_code=400, detail="image_base64 required")
    try:
        image = _decode_image(payload.image_base64)
        bbox = payload.face_bbox
        face = _face_crop(image, bbox)
        embedding = _infer_embedding_vector(face)
        if embedding is None:
            raise RuntimeError("embedding unavailable")
        return {
            "embedding": embedding.tolist(),
            "embedding_dim": int(embedding.shape[0]),
            "model": STATE.model_type,
            "model_loaded": STATE.model is not None,
        }
    except Exception as exc:
        return {
            "embedding": None,
            "embedding_dim": 0,
            "model": "face-embedder-error",
            "model_loaded": STATE.model is not None,
            "error": str(exc),
        }


@app.post("/predict/emotion")
async def predict_emotion(payload: EmotionRequest):
    print(f"[VULTUS-ML] Emotion request received for frame: {payload.frame_id or 'unknown'}")
    if not payload.image_base64:
        raise HTTPException(status_code=400, detail="image_base64 required")
    try:
        image = _decode_image(payload.image_base64)
        bbox = payload.face_bbox
        face = _face_crop(image, bbox)
        result = _infer_emotion(face)
        if result is None:
            raise RuntimeError("emotion unavailable")
        return {
            **result,
            "model": "emotion-ferplus-8",
        }
    except Exception as exc:
        return {
            "emotion": "unknown",
            "confidence": 0.0,
            "model": "emotion-detector-error",
            "error": str(exc),
        }


@app.post("/predict/face")
async def predict_face(payload: FaceRequest):
    """Full face recognition pipeline:
    1. Detect face using YOLOv8-Face
    2. Crop face with padding
    3. Generate embedding using backbone_best.pt (IResNet)
    """
    print(f"[VULTUS-ML] Face pipeline request for frame: {payload.frame_id or 'unknown'}")
    if not payload.image_base64:
        raise HTTPException(status_code=400, detail="image_base64 required")

    try:
        image = _decode_image(payload.image_base64)
    except Exception as exc:
        return {
            "person_detected": False,
            "person_confidence": 0.0,
            "embedding": None,
            "embedding_dim": 0,
            "model": "face-detector-error",
            "model_loaded": STATE.model is not None,
            "error": str(exc),
        }

    # Step 1: Detect face using YOLOv8-Face
    detection = _detect_face(image)
    if detection is None:
        return {
            "person_detected": False,
            "person_confidence": 0.0,
            "embedding": None,
            "embedding_dim": 0,
            "model": "yolov8-face",
            "model_loaded": STATE.model is not None,
        }

    x1, y1, x2, y2, score = detection

    # Step 2: Crop face with padding for embedding model
    face = _face_crop(image, (x1, y1, x2, y2))
    face_image_b64 = _encode_image_base64(face, quality=80)

    # Step 3: Generate embedding using backbone (IResNet)
    embedding = _infer_embedding_vector(face)
    if embedding is None:
        return {
            "person_detected": True,
            "person_confidence": round(score, 4),
            "person_bbox": [x1, y1, x2, y2],
            "face_image_base64": face_image_b64,
            "embedding": None,
            "embedding_dim": 0,
            "model": "face-embedder-error",
            "model_loaded": STATE.model is not None,
        }

    return {
        "person_detected": True,
        "person_confidence": round(score, 4),
        "person_bbox": [x1, y1, x2, y2],
        "face_image_base64": face_image_b64,
        "embedding": embedding.tolist(),
        "embedding_dim": int(embedding.shape[0]),
        "model": STATE.model_type,
        "model_loaded": STATE.model is not None,
    }
