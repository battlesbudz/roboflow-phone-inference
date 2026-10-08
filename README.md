# Phone Inference

A tiny, self-contained vision server that runs on a phone.

ONNX Runtime + YOLOv8 nano models and MobileNetV2 behind a six-endpoint
HTTP API. The full Roboflow inference server does this job in 7.3 GB of
dependencies and ~1.1 GB of idle RAM; this does it in ~190 MB on disk
and ~200 MB of RAM, which is the difference between "runs in a
datacenter" and "runs in Termux on your phone."

## Why

There is no official lightweight deployment story for running this class
of model on a phone. The reference server is excellent but assumes a
server. This repo is the missing small piece: a minimal server, a tested
Android install path, and honest benchmark numbers comparing the two.

It is also the vision backend for an on-device voice assistant (the phone
takes a picture, asks `POST /detect` what it sees, and talks about it),
but it stands alone: any app that can POST an image can use it.

## Quickstart

Desktop or server:

```bash
pip install -r requirements.txt
python -m uvicorn server:app --host 127.0.0.1 --port 9001
python test_client.py photo.jpg
```

Phone (Termux): see [TERMUX.md](TERMUX.md). One script installs
everything; a public project already runs this stack at 12 FPS on a 2019
mid-range phone.

## API

- `GET /health` — liveness, model identity, load time
- `POST /detect` — base64 image in, detections out
  (label, confidence, bounding box in original-image pixels)
- `POST /infer/object_detection` — same detections in Roboflow's
  request/response shapes (center x/y, width/height, class, class_id,
  detection_id), so existing Roboflow clients work with minimal changes
- `POST /infer/instance_segmentation` — boxes plus mask polygons, in
  Roboflow's instance-segmentation shapes
- `POST /infer/keypoint_detection` — person boxes plus the 17 COCO
  keypoints, in Roboflow's keypoint-detection shapes
- `POST /infer/classification` — ImageNet-1k top-5, in Roboflow's
  classification shapes

Full contract: [API.md](API.md).

Compatibility note: the `/infer/*` shapes were tested
against inference **1.7.3** (request and response classes, round-tripped
in `test_rf_compat.py`, 103 checks). That is the verified version, not a
permanent guarantee. OCR and CLIP embeddings are intentionally not
implemented (too heavy for the phone-first pass).

## Benchmarks

| | Full inference server | This repo |
|---|---|---|
| Disk | 7.3 GB | ~130 MB |
| Idle RAM | ~1.1 GB | ~160 MB |
| Cold start | 38 s | ~2 s |
| Per-frame (CPU) | ~300 ms | ~120 ms |

Methodology, raw numbers, and a blank row for your phone's results:
[BENCHMARKS.md](BENCHMARKS.md).

## Files

- `server.py` — the whole server (~750 lines)
- `termux-setup.sh` — one-shot phone bootstrap
- `yolov8n.onnx` — YOLOv8 nano, COCO 80 classes (12 MB)
- `yolov8n-seg.onnx` — YOLOv8 nano segmentation, COCO 80 classes (14 MB)
- `yolov8n-pose.onnx` — YOLOv8 nano pose, 17 COCO keypoints (14 MB)
- `mobilenetv2-12.onnx` — MobileNetV2, ImageNet-1k (14 MB)
- `synset.txt` — the 1000 ImageNet class labels
- `test-bus.jpg` — real test photo used by the compat suite
- `test_client.py` — smoke-test client
- `test_rf_compat.py` — compatibility tests for the four `/infer/*`
  endpoints (validates against the real inference 1.7.3 request/response
  classes)
- `API.md` / `BENCHMARKS.md` / `TERMUX.md` — docs

## Roadmap

- On-device benchmark numbers (in progress)
- Native Android (Kotlin + ONNX Runtime, NNAPI delegate) version to
  replace the Termux/Python layer
- Direct integration with the on-device assistant's tool loop

## License

MIT. See [LICENSE](LICENSE).
