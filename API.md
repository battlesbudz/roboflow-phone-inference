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

## Notes for integrators

- The server is single-model and stateless: every `/detect` call is
  independent. Send frames sequentially or from multiple clients; there is
  no session to manage.
- For a live camera feed, downscale frames before base64-encoding them.
  Detection quality holds up well at 640px on the long edge, and smaller
  payloads cut round-trip time more than anything else.
- Non-maximum suppression uses an IoU threshold of 0.45.
