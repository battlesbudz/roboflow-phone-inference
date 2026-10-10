# Native phone inference (Kotlin + ONNX Runtime)

Step 2 of the vision plan: the same four models as the Termux Python server
(`server.py`), running fully on-device in a native Android app — no HTTP
server, no Python, no Termux.

## Models

The `.onnx` files live at the **repo root**, shared with the Termux server —
they are not duplicated in git. At build time the `copyModels`/`copyModelExtras`
Gradle tasks copy them (plus `synset.txt` and `test-bus.jpg`) into a generated
assets directory, so the APK is self-contained:

| Asset path in APK | Endpoint parity with `server.py` |
|---|---|
| `models/yolov8n.onnx` | `Detector` — `/detect`, `/infer/object_detection` |
| `models/yolov8n-seg.onnx` | `Segmenter` — `/infer/instance_segmentation` |
| `models/yolov8n-pose.onnx` | `PoseDetector` — `/infer/keypoint_detection` |
| `models/mobilenetv2-12.onnx` | `Classifier` — `/infer/classification` |
| `synset.txt` | ImageNet-1k labels for the classifier |
| `test-bus.jpg` | Bundled benchmark image (street scene with a bus) used by the in-app sweep |

## Pre/postprocessing parity

Ported line-by-line from `server.py`:

- Letterbox to 640x640: BILINEAR resize, `(114,114,114)` canvas, centered
  paste, `/255` NCHW float tensor (`Letterbox.kt`). One known approximation:
  PIL's BILINEAR and Android's filtered `createScaledBitmap` are both bilinear
  but not bit-identical, so tensors may differ in the last decimal places.
- Greedy class-agnostic NMS, iou 0.45 default for detection (`Nms.kt`).
- Detection head: `(1, 84, 8400)` → cx,cy,w,h + 80 COCO classes; boxes mapped
  back to original-image pixels and clipped.
- Segmentation head: `(1, 116, 8400)` + `(1, 32, 160, 160)` prototypes;
  mask = sigmoid(coeffs @ protos), cropped to box/4, >0.5 threshold, scanline
  polygon trace (`Segmenter.kt`, mirrors `_scanline_polygon`).
- Pose head: `(1, 56, 8400)` = box + person score + 17×(x,y,conf); keypoints
  mapped to original-image pixels.
- Classification: direct 224x224 BILINEAR resize (no letterbox), ImageNet
  mean/std normalize, softmax top-5.

Rounding matches the server: boxes to 1 decimal, confidences to 3
(classification to 4).

## Build

CI builds a debug APK on every push to `feature/native-onnx-app`
(`.github/workflows/native-android.yml`); the APK is on the run's artifacts
page. Local build: `cd native && ./gradlew assembleDebug` (JDK 17).

## On-device benchmark

Open the app, tap **Run full sweep**. It runs the same four-endpoint sweep as
`test_all.py` on the bundled `test-bus.jpg` and prints per-endpoint
`inference_ms` (pure `session.run` time — same definition as the server's
`inference_ms` field).

Termux baselines to beat (Galaxy Z Fold 6, 2026-10-08), and native results
measured on-device (Galaxy Z Fold 6, 2026-10-09, `test-bus.jpg` 1920x1280):

| Endpoint | Termux baseline | Native |
|---|---|---|
| detection | ~285 ms | 119.0 ms (bus 0.923, car 0.898, person 0.882) |
| segmentation | ~239 ms | 150.6 ms (3 masks, first 30 polygon points) |
| pose | ~137 ms | 110.9 ms (3 persons, 17 keypoints) |
| classification | ~50 ms | 16.0 ms (streetcar/tram 0.6862) |

The first run includes one-time kernel init; the sweep does a warmup
detection first so the reported numbers are steady-state.
