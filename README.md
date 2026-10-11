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

## Install (Android app)

The native Android app (`native/`, branch `feature/native-onnx-app`) is a
standalone vision app: take or pick a photo and get a fused description from
the four on-device models, or point any local agent at its localhost HTTP
API (`GET /health`, `POST /detect` with a base64 image and the
`X-Vision-Token` header shown in the app).

- **Release APK:** download the signed APK from the
  [Releases page](../../releases) (cut from `v*` tags) and sideload it.
  No Play Store needed.
- **Debug APK:** every push to `feature/native-onnx-app` uploads
  `phone-inference-native-debug` as a workflow artifact.

## Roboflow project models

Tap **Model** in the app and enter a Roboflow API key + workspace + project
+ version to download that project's ONNX weights and use them as the
detector. The download happens once; inference runs fully offline
afterwards. The bundled stock weights (YOLOv8n etc.) stay the default so the
app works on first launch with zero setup, and the app always shows which
model is active (stock vs `workspace/project vN`).

- Only YOLO-style detection exports (YOLOv8/v11/v12) are supported; the
  project's boxes feed the same results UI, captions, and `POST /detect`
  endpoint the stock detector uses. Segmentation, pose, and classification
  always use the stock models.
- The API key is stored in EncryptedSharedPreferences (Android Keystore
  backed), never logged, and never leaves the phone except to talk to
  `api.roboflow.com`. The Model screen is FLAG_SECURE (no screenshots).
- Failures are reported in plain language: bad key, unknown
  workspace/project/version, no ONNX export for that model type, or no
  network.

Format support (per
[Roboflow's export docs](https://docs.roboflow.com/models/model-weights/download-roboflow-model-weights)):
ONNX is the recommended weights format and is available for Roboflow 3.0,
YOLOv8, YOLOv12, YOLO26, YOLO-NAS, and RF-DETR object detection plus
classification models; instance segmentation, classification, and keypoint
detection models also export to ONNX. The app resolves the download through
`GET https://api.roboflow.com/{workspace}/{project}/{version}/onnx?api_key=…`
(polling while Roboflow prepares the export) and downloads the signed
`export.link` URL, the same mechanism the official Python SDK uses for
format exports.
- **Build it yourself:** `./gradlew assembleDebug` in `native/`
  (JDK 17). Signed release builds need the keystore, which lives only in
  CI secrets — it is never committed to the repo.

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

This project is licensed under the **GNU Affero General Public License v3.0**
(AGPL-3.0) — see [LICENSE](LICENSE) for the full text.

Why AGPL: the app bundles Ultralytics YOLOv8n model weights, which are
distributed under AGPL-3.0, and that license flows through to the whole app.
In short: anyone may use, modify, and redistribute this app — including
running a modified version as a network service — as long as source changes
are shared under the same license. (Jarvis itself stays under its own
license; this app is a dependency/tool, not Jarvis.)

Component licenses:
- YOLOv8n / YOLOv8n-seg / YOLOv8n-pose weights (Ultralytics) — AGPL-3.0
- MobileNetV2 weights (ONNX Model Zoo) — Apache-2.0
- ONNX Runtime (`onnxruntime-android`) — MIT
- NanoHTTPD — BSD-3-Clause
- AndroidX (`core`, `activity`) — Apache-2.0
