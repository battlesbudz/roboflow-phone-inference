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
    POST /infer/instance_segmentation -> Roboflow-compatible instance segmentation
                                   (boxes + mask polygons).
    POST /infer/keypoint_detection    -> Roboflow-compatible keypoint detection
                                   (person boxes + 17 COCO keypoints).
    POST /infer/classification        -> Roboflow-compatible classification
                                   (ImageNet-1k top-5).

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
VERSION = "1.2.0"

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


def _letterbox(image: Image.Image) -> tuple[np.ndarray, float, int, int, int, int]:
    """Letterbox to 640x640. Returns (tensor, scale, pad_x, pad_y, orig_w, orig_h)."""
    orig_w, orig_h = image.size
    scale = min(INPUT_SIZE / orig_w, INPUT_SIZE / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (114, 114, 114))
    pad_x, pad_y = (INPUT_SIZE - new_w) // 2, (INPUT_SIZE - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    x = np.array(canvas).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return x, scale, pad_x, pad_y, orig_w, orig_h


def _xyxy_from_cxcywh(boxes_cxcywh: np.ndarray) -> np.ndarray:
    """cxcywh -> xyxy (same coordinate space)."""
    xyxy = np.zeros_like(boxes_cxcywh)
    xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
    return xyxy


def _map_to_original(
    xyxy: np.ndarray, scale: float, pad_x: int, pad_y: int, orig_w: int, orig_h: int
) -> np.ndarray:
    """Map letterboxed xyxy boxes back to original-image pixels, clipped."""
    out = xyxy.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - pad_x) / scale
    out[:, [1, 3]] = (out[:, [1, 3]] - pad_y) / scale
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, orig_w)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, orig_h)
    return out


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
        x, scale, pad_x, pad_y, orig_w, orig_h = _letterbox(image)

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
        xyxy = _map_to_original(
            _xyxy_from_cxcywh(boxes_cxcywh), scale, pad_x, pad_y, orig_w, orig_h
        )

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
    models = {}
    for name, obj, fname in (
        ("object_detection", detector, "yolov8n.onnx"),
        ("instance_segmentation", segmenter, "yolov8n-seg.onnx"),
        ("keypoint_detection", pose_detector, "yolov8n-pose.onnx"),
        ("classification", classifier, "mobilenetv2-12.onnx"),
    ):
        models[name] = {"file": fname, "load_ms": round(obj.load_ms, 1)}
    return {
        "status": "ok",
        "model": "yolov8n",
        "version": VERSION,
        "model_load_ms": round(detector.load_ms, 1),
        "models": models,
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


# ---------------------------------------------------------------------------
# Instance segmentation, keypoint detection, classification.
#
# Same Roboflow-compatibility approach as /infer/object_detection above,
# validated against the inference 1.7.3 request/response entity classes:
#   InstanceSegmentationInferenceRequest, KeypointsDetectionInferenceRequest,
#   ClassificationInferenceRequest, plus the InstanceSegmentation /
#   Keypoints / Classification response shapes.
#
# Real models, no stubs:
#   - yolov8n-seg.onnx  (YOLOv8n-seg, COCO 80 classes): boxes + mask polygons
#   - yolov8n-pose.onnx (YOLOv8n-pose, 17 COCO keypoints): boxes + keypoints
#   - mobilenetv2-12.onnx (ImageNet-1k): top-5 classification
#
# OCR and CLIP embeddings are also standard Roboflow endpoints but are too
# heavy for this phone-first pass and are intentionally not implemented.
# ---------------------------------------------------------------------------

SEG_MODEL_PATH = str(Path(__file__).resolve().parent / "yolov8n-seg.onnx")
POSE_MODEL_PATH = str(Path(__file__).resolve().parent / "yolov8n-pose.onnx")
CLS_MODEL_PATH = str(Path(__file__).resolve().parent / "mobilenetv2-12.onnx")
SYNSET_PATH = str(Path(__file__).resolve().parent / "synset.txt")

COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLS_INPUT_SIZE = 224


def _load_imagenet_labels() -> list[str]:
    labels = []
    with open(SYNSET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # "n01440764 tench, Tinca tinca" -> "tench, Tinca tinca"
            labels.append(line.split(" ", 1)[1] if " " in line else line)
    return labels


CLS_LABELS = _load_imagenet_labels()


def _scanline_polygon(binm: np.ndarray, max_points: int = 200) -> list[tuple[float, float]]:
    """Outline a binary mask as a polygon, in mask-pixel coordinates.

    Marching squares needs OpenCV/skimage, both too heavy for the phone
    build, so this traces the mask row by row: down the left edge, back up
    the right edge. The result is a valid (possibly blocky) polygon that
    follows the mask outline. Holes are ignored.
    """
    rows = np.flatnonzero(binm.any(axis=1))
    if len(rows) == 0:
        return []
    left, right = [], []
    for y in rows:
        xs = np.flatnonzero(binm[int(y)])
        left.append((float(xs[0]), float(y)))
        right.append((float(xs[-1]), float(y)))
    poly = left + right[::-1]
    if len(poly) > max_points:
        step = len(poly) / max_points
        poly = [poly[int(k * step)] for k in range(max_points)]
    return poly


class Segmenter:
    def __init__(self, model_path: str = SEG_MODEL_PATH):
        t0 = time.time()
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.load_ms = (time.time() - t0) * 1000

    def segment(
        self,
        image: Image.Image,
        confidence: float = 0.4,
        iou_threshold: float = 0.3,
        class_filter: Optional[list] = None,
        max_detections: int = 300,
    ) -> tuple[list[dict], float]:
        """Returns ([{"box_xyxy", "confidence", "label", "class_id", "points"}], infer_ms).

        Points are mask-polygon vertices in ORIGINAL image pixels.
        """
        x, scale, pad_x, pad_y, orig_w, orig_h = _letterbox(image)

        t0 = time.time()
        out0, protos = self.session.run(None, {self.input_name: x})
        infer_ms = (time.time() - t0) * 1000

        # out0: (1, 116, 8400) = 4 box + 80 classes + 32 mask coefficients
        # protos: (1, 32, 160, 160) mask prototypes at 1/4 of the 640 frame
        preds = out0[0].transpose()  # (8400, 116)
        boxes_cxcywh = preds[:, :4]
        scores = preds[:, 4:84].max(axis=1)
        labels = preds[:, 4:84].argmax(axis=1)
        coeffs = preds[:, 84:116]
        keep = scores >= confidence
        boxes_cxcywh, scores, labels, coeffs = (
            boxes_cxcywh[keep],
            scores[keep],
            labels[keep],
            coeffs[keep],
        )

        xyxy_lb = _xyxy_from_cxcywh(boxes_cxcywh)  # letterboxed 640 coords
        xyxy = _map_to_original(xyxy_lb, scale, pad_x, pad_y, orig_w, orig_h)

        proto = protos[0]  # (32, 160, 160)
        results = []
        for i in _nms(xyxy, scores, iou_thresh=iou_threshold):
            label = COCO_LABELS[int(labels[i])]
            if class_filter is not None and label not in class_filter:
                continue
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])

            # assemble the mask: sigmoid(coeffs @ prototypes), crop to box
            raw = 1.0 / (1.0 + np.exp(-(coeffs[i] @ proto.reshape(32, -1))))
            mask160 = raw.reshape(160, 160)
            bx1, by1, bx2, by2 = (xyxy_lb[i] / 4).astype(int)
            bx1, by1 = max(bx1, 0), max(by1, 0)
            bx2, by2 = min(bx2, 160), min(by2, 160)
            points = []
            if bx2 > bx1 and by2 > by1:
                crop = (mask160[by1:by2, bx1:bx2] > 0.5)
                poly = _scanline_polygon(crop)
                for mx, my in poly:
                    # proto coords -> letterboxed 640 coords -> original pixels
                    lx, ly = (mx + bx1) * 4.0, (my + by1) * 4.0
                    ox = min(max((lx - pad_x) / scale, 0.0), float(orig_w))
                    oy = min(max((ly - pad_y) / scale, 0.0), float(orig_h))
                    points.append({"x": round(ox, 1), "y": round(oy, 1)})

            results.append(
                {
                    "box_xyxy": (round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)),
                    "confidence": round(float(scores[i]), 3),
                    "label": label,
                    "class_id": int(labels[i]),
                    "points": points,
                }
            )
        results.sort(key=lambda r: r["confidence"], reverse=True)
        return results[:max_detections], infer_ms


class PoseDetector:
    def __init__(self, model_path: str = POSE_MODEL_PATH):
        t0 = time.time()
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.load_ms = (time.time() - t0) * 1000

    def pose(
        self,
        image: Image.Image,
        confidence: float = 0.4,
        iou_threshold: float = 0.3,
        keypoint_confidence: float = 0.0,
        class_filter: Optional[list] = None,
        max_detections: int = 300,
    ) -> tuple[list[dict], float]:
        """Returns ([{"box_xyxy", "confidence", "keypoints": [{x,y,confidence,
        class, class_id}]}], infer_ms). Keypoints in ORIGINAL image pixels."""
        x, scale, pad_x, pad_y, orig_w, orig_h = _letterbox(image)

        t0 = time.time()
        out = self.session.run(None, {self.input_name: x})[0]
        infer_ms = (time.time() - t0) * 1000

        # (1, 56, 8400) = 4 box + 1 person score + 17 keypoints * (x, y, conf)
        preds = out[0].transpose()  # (8400, 56)
        boxes_cxcywh = preds[:, :4]
        scores = preds[:, 4]
        kpts = preds[:, 5:].reshape(-1, 17, 3)
        keep = scores >= confidence
        boxes_cxcywh, scores, kpts = boxes_cxcywh[keep], scores[keep], kpts[keep]

        xyxy = _map_to_original(
            _xyxy_from_cxcywh(boxes_cxcywh), scale, pad_x, pad_y, orig_w, orig_h
        )

        results = []
        for i in _nms(xyxy, scores, iou_thresh=iou_threshold):
            if class_filter is not None and "person" not in class_filter:
                continue
            x1, y1, x2, y2 = (round(float(v), 1) for v in xyxy[i])
            keypoints = []
            for j, (kx, ky, kc) in enumerate(kpts[i]):
                if float(kc) < keypoint_confidence:
                    continue
                ox = min(max((float(kx) - pad_x) / scale, 0.0), float(orig_w))
                oy = min(max((float(ky) - pad_y) / scale, 0.0), float(orig_h))
                keypoints.append(
                    {
                        "x": round(ox, 1),
                        "y": round(oy, 1),
                        "confidence": round(float(kc), 3),
                        "class": COCO_KEYPOINT_NAMES[j],
                        "class_id": j,
                    }
                )
            results.append(
                {
                    "box_xyxy": (x1, y1, x2, y2),
                    "confidence": round(float(scores[i]), 3),
                    "keypoints": keypoints,
                }
            )
        results.sort(key=lambda r: r["confidence"], reverse=True)
        return results[:max_detections], infer_ms


class Classifier:
    def __init__(self, model_path: str = CLS_MODEL_PATH):
        t0 = time.time()
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.load_ms = (time.time() - t0) * 1000

    def classify(self, image: Image.Image) -> tuple[list[dict], float]:
        """ImageNet-1k top-5. Returns ([{"class", "class_id", "confidence"}], infer_ms)."""
        resized = image.resize((CLS_INPUT_SIZE, CLS_INPUT_SIZE), Image.BILINEAR)
        x = np.array(resized).astype(np.float32) / 255.0
        x = ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]

        t0 = time.time()
        logits = self.session.run(None, {self.input_name: x})[0][0]
        infer_ms = (time.time() - t0) * 1000

        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        top5 = probs.argsort()[::-1][:5]
        return (
            [
                {
                    "class": CLS_LABELS[int(k)],
                    "class_id": int(k),
                    "confidence": round(float(probs[k]), 4),
                }
                for k in top5
            ],
            infer_ms,
        )


segmenter = Segmenter()
pose_detector = PoseDetector()
classifier = Classifier()


# ---------------------------------------------------------------------------
# Roboflow-compatible request/response models for the three new endpoints.
# ---------------------------------------------------------------------------


class RFBaseInferRequest(BaseModel):
    image: RFImage
    api_key: Optional[str] = None  # accepted, ignored: no hosted account
    model_id: Optional[str] = None  # accepted, ignored: single local model
    confidence: Union[float, str] = 0.4  # float 0..1, or "best"/"default"
    iou_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    class_filter: Optional[list[str]] = None
    max_detections: int = Field(default=300, ge=1)


class RFSegInferRequest(RFBaseInferRequest):
    # accepted and ignored: this server always decodes masks the same way
    mask_decode_mode: str = "accurate"


class RFPoseInferRequest(RFBaseInferRequest):
    keypoint_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # accepted and ignored: per-keypoint NMS is not implemented
    keypoint_iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class RFClassInferRequest(BaseModel):
    image: RFImage
    api_key: Optional[str] = None  # accepted, ignored
    model_id: Optional[str] = None  # accepted, ignored
    confidence: Union[float, str] = 0.4  # min top-1 to report as `top`


class RFPoint(BaseModel):
    x: float
    y: float


class RFSegPrediction(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_name: str = Field(serialization_alias="class")
    class_id: int
    detection_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    points: list[RFPoint]


class RFKeypoint(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    x: float
    y: float
    confidence: float
    class_id: int
    class_name: str = Field(serialization_alias="class")


class RFPosePrediction(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_name: str = Field(serialization_alias="class")
    class_id: int
    detection_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    keypoints: list[RFKeypoint]


class RFClassPrediction(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(serialization_alias="class")
    class_id: int
    confidence: float


class RFClassResponse(BaseModel):
    predictions: list[RFClassPrediction]
    top: str
    confidence: float
    image: RFImageInfo
    inference_id: str
    time: float


class RFSegInferResponse(BaseModel):
    predictions: list[RFSegPrediction]
    image: RFImageInfo
    inference_id: str
    time: float


class RFPoseInferResponse(BaseModel):
    predictions: list[RFPosePrediction]
    image: RFImageInfo
    inference_id: str
    time: float


def _resolve_pose_confidence(confidence: Union[float, str]) -> float:
    # parity with inference 1.7.3: "best" is rejected for keypoint models
    if confidence == "best":
        raise HTTPException(
            status_code=400,
            detail='confidence="best" is not supported for keypoint detection; use a float or "default"',
        )
    return _resolve_confidence(confidence)


@app.post("/infer/instance_segmentation", response_model=RFSegInferResponse)
def infer_instance_segmentation(req: RFSegInferRequest):
    confidence = _resolve_confidence(req.confidence)
    image = _load_rf_image(req.image)
    orig_w, orig_h = image.size
    results, _infer_ms = segmenter.segment(
        image,
        confidence,
        iou_threshold=req.iou_threshold,
        class_filter=req.class_filter,
        max_detections=req.max_detections,
    )
    predictions = []
    for r in results:
        x1, y1, x2, y2 = r["box_xyxy"]
        predictions.append(
            RFSegPrediction(
                x=round((x1 + x2) / 2, 1),
                y=round((y1 + y2) / 2, 1),
                width=round(x2 - x1, 1),
                height=round(y2 - y1, 1),
                confidence=r["confidence"],
                class_name=r["label"],
                class_id=r["class_id"],
                points=[RFPoint(x=p["x"], y=p["y"]) for p in r["points"]],
            )
        )
    return RFSegInferResponse(
        predictions=predictions,
        image=RFImageInfo(width=orig_w, height=orig_h),
        inference_id=str(uuid.uuid4()),
        time=time.time(),
    )


@app.post("/infer/keypoint_detection", response_model=RFPoseInferResponse)
def infer_keypoint_detection(req: RFPoseInferRequest):
    confidence = _resolve_pose_confidence(req.confidence)
    image = _load_rf_image(req.image)
    orig_w, orig_h = image.size
    results, _infer_ms = pose_detector.pose(
        image,
        confidence,
        iou_threshold=req.iou_threshold,
        keypoint_confidence=req.keypoint_confidence,
        class_filter=req.class_filter,
        max_detections=req.max_detections,
    )
    predictions = []
    for r in results:
        x1, y1, x2, y2 = r["box_xyxy"]
        predictions.append(
            RFPosePrediction(
                x=round((x1 + x2) / 2, 1),
                y=round((y1 + y2) / 2, 1),
                width=round(x2 - x1, 1),
                height=round(y2 - y1, 1),
                confidence=r["confidence"],
                class_name="person",
                class_id=0,
                keypoints=[
                    RFKeypoint(
                        x=k["x"],
                        y=k["y"],
                        confidence=k["confidence"],
                        class_id=k["class_id"],
                        class_name=k["class"],
                    )
                    for k in r["keypoints"]
                ],
            )
        )
    return RFPoseInferResponse(
        predictions=predictions,
        image=RFImageInfo(width=orig_w, height=orig_h),
        inference_id=str(uuid.uuid4()),
        time=time.time(),
    )


@app.post("/infer/classification")
def infer_classification(req: RFClassInferRequest):
    confidence = _resolve_confidence(req.confidence)
    image = _load_rf_image(req.image)
    orig_w, orig_h = image.size
    top5, _infer_ms = classifier.classify(image)
    top = top5[0]
    return RFClassResponse(
        predictions=[
            RFClassPrediction(
                class_name=p["class"], class_id=p["class_id"], confidence=p["confidence"]
            )
            for p in top5
        ],
        top=top["class"] if top["confidence"] >= confidence else "",
        confidence=top["confidence"] if top["confidence"] >= confidence else 0.0,
        image=RFImageInfo(width=orig_w, height=orig_h),
        inference_id=str(uuid.uuid4()),
        time=time.time(),
    )
