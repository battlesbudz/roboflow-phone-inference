# API

Base URL: `http://localhost:9001` (default; configurable via uvicorn flags).

All responses are JSON.

## GET /health

Liveness check. Returns model identity and how long the model took to load.

Response:

```json
{
  "status": "ok",
  "model": "yolov8n",
  "version": "1.2.0",
  "model_load_ms": 156.6,
  "models": {
    "object_detection": {"file": "yolov8n.onnx", "load_ms": 156.6},
    "instance_segmentation": {"file": "yolov8n-seg.onnx", "load_ms": 98.2},
    "keypoint_detection": {"file": "yolov8n-pose.onnx", "load_ms": 91.4},
    "classification": {"file": "mobilenetv2-12.onnx", "load_ms": 42.0}
  }
}
```

## POST /detect

Run object detection on one image.

Request body:

```json
{
  "image": "<base64-encoded PNG or JPEG>",
  "confidence": 0.25
}
```

- `image` (required): base64-encoded PNG or JPEG bytes.
- `confidence` (optional, default 0.25): minimum confidence score for a
  detection to be returned. Range 0.0 to 1.0.

Response:

```json
{
  "detections": [
    {
      "label": "person",
      "confidence": 0.871,
      "box": {"x1": 112.4, "y1": 88.0, "x2": 246.9, "y2": 402.5}
    }
  ],
  "inference_ms": 116.7
}
```

- `label`: one of the 80 COCO class names (see `server.py`, `COCO_LABELS`).
- `confidence`: 0.0 to 1.0.
- `box`: pixel coordinates in the *original* image (before the internal
  640x640 letterbox resize). `x1,y1` is the top-left corner, `x2,y2` the
  bottom-right.
- `inference_ms`: server-side model time for this frame. Excludes network,
  base64, and image decode.

Errors:

- `400` — `image` is not valid base64, or the bytes are not a decodable image.
- `422` — request body is not valid JSON or is missing the `image` field.

## POST /infer/object_detection

Roboflow-compatible object detection. Same model, same boxes, wrapped in
the request/response shapes of the Roboflow inference server, so existing
Roboflow clients and code written against that API work with minimal
changes.

Compatibility was tested against inference **1.7.3**. That is the version
this was verified with, not a promise of permanent parity: if Roboflow
changes its API, this endpoint may need updating. Scope is object
detection; see below for the segmentation, keypoint, and classification
endpoints.

Request body:

```json
{
  "image": {"type": "base64", "value": "<base64-encoded PNG or JPEG>"},
  "confidence": 0.4,
  "iou_threshold": 0.3,
  "class_filter": ["person", "car"],
  "max_detections": 300,
  "api_key": "anything",
  "model_id": "anything"
}
```

- `image` (required): `{"type": "base64", "value": ...}` or
  `{"type": "url", "value": "https://..."}`. A `data:` URL prefix on a
  base64 value is tolerated. Type `"numpy"` and batch lists are rejected
  with `400`.
- `confidence` (optional, default 0.4): float 0.0 to 1.0, or `"best"` /
  `"default"` (both map to 0.4; there are no per-model eval thresholds
  here).
- `iou_threshold` (optional, default 0.3): NMS IoU threshold.
- `class_filter` (optional): only predictions with these labels are
  returned. Labels are the 80 COCO class names.
- `max_detections` (optional, default 300): cap on returned predictions.
- `api_key`, `model_id` (optional): accepted and ignored. This server runs
  one local model and has no hosted account to bill.
- Other Roboflow options (`visualize_predictions`, `disable_preproc_*`,
  `class_agnostic_nms`, etc.) are accepted and ignored.

Response:

```json
{
  "predictions": [
    {
      "x": 179.6,
      "y": 245.2,
      "width": 134.5,
      "height": 314.5,
      "confidence": 0.871,
      "class": "person",
      "class_id": 0,
      "detection_id": "3f9a2c1e-..."
    }
  ],
  "image": {"width": 640, "height": 480},
  "inference_id": "9b1e4f2a-...",
  "time": 1728278400.0
}
```

- `x`, `y`: center of the box in original-image pixels. `width`,
  `height`: box size in pixels.
- `class`: COCO label. `class_id`: its index in the COCO list.
- `detection_id`: random UUID per prediction, same convention as the
  reference server.
- `image`: original image dimensions. `time`: Unix epoch of the request.

Errors:

- `400` — bad base64, undecodable image, unreachable URL, unsupported
  `image.type`, or an invalid `confidence` value.
- `422` — body is not valid JSON or is missing the `image` field.

Simplifications vs the reference server, stated plainly:

- NMS is class-agnostic (one pass over all boxes). Per-class NMS is not
  implemented.
- `visualize_predictions` is ignored; no annotated image is returned.
- One model only. There is no model registry, so `model/add`, `remove`,
  and `clear` do not exist here.

## POST /infer/instance_segmentation

Roboflow-compatible instance segmentation. Runs the real
`yolov8n-seg.onnx` model (YOLOv8 nano segmentation, COCO 80 classes) and
returns boxes plus a mask polygon per instance.

Request body: same fields as `/infer/object_detection`, plus
`mask_decode_mode` (optional, default `"accurate"`; accepted and ignored —
this server always decodes masks the same way).

Response:

```json
{
  "predictions": [
    {
      "x": 412.5,
      "y": 483.1,
      "width": 775.8,
      "height": 507.2,
      "confidence": 0.858,
      "class": "bus",
      "class_id": 5,
      "detection_id": "3f9a2c1e-...",
      "points": [
        {"x": 202.5, "y": 229.5},
        {"x": 189.0, "y": 236.2}
      ]
    }
  ],
  "image": {"width": 810, "height": 1080},
  "inference_id": "9b1e4f2a-...",
  "time": 1728278400.0
}
```

- `points`: mask polygon vertices in original-image pixels (up to 200
  points; may be fewer for small masks, empty if the mask vanished after
  thresholding). Holes are not represented.
- Boxes, `x`/`y` centers, and `image` follow the same conventions as
  `/infer/object_detection`.

Errors: same as `/infer/object_detection`.

## POST /infer/keypoint_detection

Roboflow-compatible keypoint detection. Runs the real
`yolov8n-pose.onnx` model (YOLOv8 nano pose; detects people and their 17
COCO keypoints) and returns person boxes plus keypoints.

Request body: same fields as `/infer/object_detection`, plus:

- `keypoint_confidence` (optional, default 0.0): keypoints below this
  confidence are dropped from the response.
- `keypoint_iou_threshold` (optional, default 0.5): accepted and
  ignored; per-keypoint NMS is not implemented.

Note: `confidence: "best"` is rejected with `400`, matching the
reference server (1.7.3) which does not produce per-class thresholds for
keypoint models. `"default"` maps to 0.4.

Response:

```json
{
  "predictions": [
    {
      "x": 166.7,
      "y": 385.8,
      "width": 115.9,
      "height": 301.8,
      "confidence": 0.883,
      "class": "person",
      "class_id": 0,
      "detection_id": "3f9a2c1e-...",
      "keypoints": [
        {"x": 143.1, "y": 441.9, "confidence": 0.98,
         "class": "nose", "class_id": 0}
      ]
    }
  ],
  "image": {"width": 810, "height": 1080},
  "inference_id": "9b1e4f2a-...",
  "time": 1728278400.0
}
```

- `keypoints`: one entry per visible keypoint (17 COCO keypoints:
  nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles),
  in original-image pixels, with per-keypoint confidence.

Errors: same as `/infer/object_detection`, plus `400` for
`confidence: "best"`.

## POST /infer/classification

Roboflow-compatible image classification. Runs the real
`mobilenetv2-12.onnx` model (MobileNetV2, ImageNet-1k) and returns the
top-5 predictions.

Request body:

```json
{
  "image": {"type": "base64", "value": "<base64-encoded PNG or JPEG>"},
  "confidence": 0.4,
  "api_key": "anything",
  "model_id": "anything"
}
```

- `confidence` (optional, default 0.4): minimum top-1 confidence for
  `top` to be reported. The `predictions` top-5 list is returned
  regardless; below the threshold, `top` is `""` and `confidence` is
  `0.0`.
- `api_key`, `model_id` (optional): accepted and ignored.

Response:

```json
{
  "predictions": [
    {"class": "minibus", "class_id": 779, "confidence": 0.737},
    {"class": "police van, police wagon, police patrol wagon, paddy wagon, patrol wagon", "class_id": 573, "confidence": 0.2016}
  ],
  "top": "minibus",
  "confidence": 0.737,
  "image": {"width": 810, "height": 1080},
  "inference_id": "9b1e4f2a-...",
  "time": 1728278400.0
}
```

- `top` / `confidence`: the top-1 label and its confidence (gated by the
  request's `confidence` as described above).
- `predictions`: top-5, sorted by confidence descending.

Errors:

- `400` — bad base64, undecodable image, unreachable URL, unsupported
  `image.type`, or an invalid `confidence` value.
- `422` — body is not valid JSON or is missing the `image` field.

## Notes for integrators

- The server is single-model and stateless: every call is independent.
  Send frames sequentially or from multiple clients; there is no session
  to manage.
- For a live camera feed, downscale frames before base64-encoding them.
  Detection quality holds up well at 640px on the long edge, and smaller
  payloads cut round-trip time more than anything else.
- Non-maximum suppression uses an IoU threshold of 0.45 on `/detect`.
  The three `/infer/*` endpoints default to 0.3, matching the reference
  server, and accept an `iou_threshold` per request.
- OCR and CLIP embeddings are standard Roboflow endpoints but are too
  heavy for the phone-first pass and are intentionally not implemented.
