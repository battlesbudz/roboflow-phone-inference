"""
Lean inference server for phones (Termux-friendly).

A tiny HTTP wrapper around ONNX Runtime + a YOLOv8 nano model.
Same job as the full Roboflow inference server, none of the 7GB baggage.

API:
    GET  /health                -> {"status": "ok", "model": ..., "version": ...}
    POST /detect                -> {"image": "<base64 png/jpg>", "confidence": 0.25 (optional)}
                                   returns {"detections": [{"label", "confidence", "box": {"x1","y1","x2","y2"}}],
                                            "inference_ms": ...}
    POST /infer/object_detection -> Roboflow-compatible object detection.
                                   Accepts {"image": {"type": "base64"|"url", "value": ...},
                                            "confidence", "iou_threshold",
                                            "class_filter", "max_detections",
                                            "api_key", "model_id"}.
                                   Returns {"predictions": [{"x","y","width","height",
                                            "confidence","class","class_id","detection_id"}],
                                            "image": {"width","height"}, ...}.

Boxes are in the ORIGINAL image's pixel coordinates.
"""

import base64
import binascii
import io
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Optional, Union

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = str(Path(__file__).resolve().parent / "yolov8n.onnx")
INPUT_SIZE = 640
VERSION = "1.1.0"

COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


class DetectRequest(BaseModel):
    image: str  # base64-encoded PNG or JPEG
    confidence: float = 0.25


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    label: str
    confidence: float
    box: Box


class DetectResponse(BaseModel):
    detections: list[Detection]
    inference_ms: float


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.45) -> list[int]:
    """Greedy NMS. boxes: (N,4) xyxy, scores: (N,). Returns kept indices."""
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (
            boxes[order[1:], 3] - boxes[order[1:], 1]
        )
        iou = inter / (area_i + area_rest - inter + 1e-6)
        order = order[1:][iou <= iou_thresh]
    return keep


class Detector:
    def __init__(self, model_path: str = MODEL_PATH):
        t0 = time.time()
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.load_ms = (time.time() - t0) * 1000

    def detect(
        self,
        image: Image.Image,
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        class_filter: Optional[list] = None,
        max_detections: Optional[int] = None,
    ) -> tuple[list[Detection], float]:
        orig_w, orig_h = image.size
        # letterbox to 640x640
        scale = min(INPUT_SIZE / orig_w, INPUT_SIZE / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        resized = image.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (114, 114, 114))
        pad_x, pad_y = (INPUT_SIZE - new_w) // 2, (INPUT_SIZE - new_h) // 2
        canvas.paste(resized, (pad_x, pad_y))
        x = np.array(canvas).astype(np.float32).transpose(2, 0, 1)[None] / 255.0

        t0 = time.time()
        out = self.session.run(None, {self.input_name: x})[0]  # (1, 84, 8400)
        infer_ms = (time.time() - t0) * 1000

        preds = out[0].transpose()  # (8400, 84): cx,cy,w,h + 80 classes
        boxes_cxcywh = preds[:, :4]
        scores = preds[:, 4:].max(axis=1)
        labels = preds[:, 4:].argmax(axis=1)
        mask = scores >= confidence
        boxes_cxcywh, scores, labels = boxes_cxcywh[mask], scores[mask], labels[mask]

        # cxcywh -> xyxy (in letterboxed coords), then map back to original
        xyxy = np.zeros_like(boxes_cxcywh)
        xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / scale
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / scale
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, orig_w)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, orig_h)

        detections = []
        for i in _nms(xyxy, scores, iou_thresh=iou_threshold):
            label = COCO_LABELS[int(labels[i])]
            if class_filter is not None and label not in class_filter:
                continue
            x1, y1, x2, y2 = (round(float(v), 1) for v in xyxy[i])
            detections.append(
                Detection(
                    label=label,
                    confidence=round(float(scores[i]), 3),
                    box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        if max_detections is not None:
            detections = detections[:max_detections]
        return detections, infer_ms


detector = Detector()
app = FastAPI(title="Jarvis Lean Inference Server", version=VERSION)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "yolov8n",
        "version": VERSION,
        "model_load_ms": round(detector.load_ms, 1),
    }


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    try:
        raw = base64.b64decode(req.image)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="image is not valid base64")
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="image could not be decoded as PNG/JPEG")
    detections, infer_ms = detector.detect(image, req.confidence)
    return DetectResponse(detections=detections, inference_ms=round(infer_ms, 1))


# ---------------------------------------------------------------------------
# Roboflow-compatible object detection.
#
# Tested against inference 1.7.3 request/response shapes. This is a
# compatibility layer, not a promise of permanent parity: Roboflow's server
# may change its API, and this endpoint covers object detection only.
#
# Simplifications vs the full server:
#   - NMS is class-agnostic (one pass over all boxes). Per-class NMS
#     (class_agnostic_nms=false) is not implemented.
#   - confidence "best"/"default" map to the 0.4 default; there are no
#     per-model eval thresholds here.
#   - api_key and model_id are accepted and ignored. This server runs one
#     local model and has no hosted account to bill.
#   - visualize_predictions and the other visualization/* and
#     disable_preproc_* options are accepted and ignored.
#   - image as a list (batch) and type "numpy" are rejected with 400.
# ---------------------------------------------------------------------------


class RFImage(BaseModel):
    type: str  # "base64" or "url"
    value: str


class RFInferRequest(BaseModel):
    image: RFImage
    api_key: Optional[str] = None  # accepted, ignored: no hosted account
    model_id: Optional[str] = None  # accepted, ignored: single local model
    confidence: Union[float, str] = 0.4  # float 0..1, or "best"/"default"
    iou_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    class_filter: Optional[list[str]] = None
    max_detections: int = Field(default=300, ge=1)


class RFPrediction(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    x: float  # center x, original-image pixels
    y: float  # center y, original-image pixels
    width: float
    height: float
    confidence: float
    class_name: str = Field(serialization_alias="class")
    class_id: int
    detection_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class RFImageInfo(BaseModel):
    width: int
    height: int


class RFInferResponse(BaseModel):
    predictions: list[RFPrediction]
    image: RFImageInfo
    inference_id: str
    time: float


def _resolve_confidence(confidence: Union[float, str]) -> float:
    if isinstance(confidence, str):
        if confidence in ("best", "default"):
            return 0.4
        raise HTTPException(
            status_code=400,
            detail='confidence must be a float 0..1, "best", or "default"',
        )
    if not 0.0 <= confidence <= 1.0:
        raise HTTPException(
            status_code=400, detail="confidence must be between 0.0 and 1.0"
        )
    return confidence


def _load_rf_image(img: RFImage) -> Image.Image:
    if img.type == "base64":
        value = img.value
        if value.startswith("data:"):  # tolerate data URLs
            value = value.split(",", 1)[1] if "," in value else ""
        try:
            raw = base64.b64decode(value)
        except (binascii.Error, ValueError):
            raise HTTPException(status_code=400, detail="image.value is not valid base64")
    elif img.type == "url":
        try:
            req = urllib.request.Request(
                img.value, headers={"User-Agent": f"phone-inference/{VERSION}"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read(10_000_000)  # 10 MB cap
        except Exception:
            raise HTTPException(status_code=400, detail="could not fetch image from url")
    else:
        raise HTTPException(
            status_code=400,
            detail='image.type must be "base64" or "url"',
        )
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400, detail="image could not be decoded as PNG/JPEG"
        )


@app.post("/infer/object_detection", response_model=RFInferResponse)
def infer_object_detection(req: RFInferRequest):
    confidence = _resolve_confidence(req.confidence)
    image = _load_rf_image(req.image)
    orig_w, orig_h = image.size
    detections, _infer_ms = detector.detect(
        image,
        confidence,
        iou_threshold=req.iou_threshold,
        class_filter=req.class_filter,
        max_detections=req.max_detections,
    )
    predictions = []
    for d in detections:
        x1, y1, x2, y2 = d.box.x1, d.box.y1, d.box.x2, d.box.y2
        predictions.append(
            RFPrediction(
                x=round((x1 + x2) / 2, 1),
                y=round((y1 + y2) / 2, 1),
                width=round(x2 - x1, 1),
                height=round(y2 - y1, 1),
                confidence=d.confidence,
                class_name=d.label,
                class_id=COCO_LABELS.index(d.label),
            )
        )
    return RFInferResponse(
        predictions=predictions,
        image=RFImageInfo(width=orig_w, height=orig_h),
        inference_id=str(uuid.uuid4()),
        time=time.time(),
    )
