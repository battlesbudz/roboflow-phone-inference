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
  "version": "1.0.0",
  "model_load_ms": 156.6
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
detection only. Classification, segmentation, keypoints, embeddings, and
workflows are not covered.

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

## Notes for integrators

- The server is single-model and stateless: every call is independent.
  Send frames sequentially or from multiple clients; there is no session
  to manage.
- For a live camera feed, downscale frames before base64-encoding them.
  Detection quality holds up well at 640px on the long edge, and smaller
  payloads cut round-trip time more than anything else.
- Non-maximum suppression uses an IoU threshold of 0.45 on `/detect`.
  `/infer/object_detection` defaults to 0.3, matching the reference
  server, and accepts an `iou_threshold` per request.
