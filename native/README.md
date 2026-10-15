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
| `models/yolov8n.onnx` | `Detector` — `/detect`, `/infer/object_detection` (stock fallback) |
| `models/yolov8n-seg.onnx` | `Segmenter` — `/infer/instance_segmentation` |
| `models/yolov8n-pose.onnx` | `PoseDetector` — `/infer/keypoint_detection` |
| `models/mobilenetv2-12.onnx` | `Classifier` — `/infer/classification` |
| `models/rf-detr-nano.onnx` | `RfDetrDetector` — the DEFAULT detector (see below) |
| `synset.txt` | ImageNet-1k labels for the classifier |
| `test-bus.jpg` | Bundled benchmark image (street scene with a bus) used by the in-app sweep |

## RF-DETR Nano (Roboflow) — the bundled default detector

`rf-detr-nano.onnx` is Roboflow's own object-detection architecture
(RF-DETR Nano), exported with Roboflow's `rfdetr` code from the official
COCO-pretrained checkpoint, then dynamically quantized to int8 with ONNX
Runtime's `quantize_dynamic` (MatMul/Gemm/Attention weights to int8; Conv
layers stay fp32). Full provenance — export command, versions, SHA-256,
parity results — lives in `RFDETR_PROVENANCE.json` at the repo root.

Why dynamic quantization: `rfdetr`'s `export()` quantization parameter is
TFLite-only and silently ignored for ONNX, so the exporter only produces a
~115 MB fp32 file (over GitHub's 100 MB per-file limit). The fp16 converter
produces a graph ONNX Runtime rejects. Dynamic quantization is ORT's own
tool, so the output loads by construction, and it brings the file well under
the limit. Export-time parity gate: same top-5 label set as fp32, scores
within 0.1, bus detected on `test-bus.jpg`.

Decode contract (mirrors Roboflow's own runtime): direct resize to the
model's input size (no letterbox), /255, ImageNet mean/std, NCHW; outputs
`dets` (1, Q, 4) normalised cxcywh + `labels` (1, Q, 91) raw logits;
per-class sigmoid, top-Q pairs before thresholding, score > 0.3, no NMS;
slot index = COCO category ID (sparse 1-90). `RfDetrDetector.kt` implements
it; `verify_rfdetr.py` checks the same contract in CI.

The Model screen switches between RF-DETR (bundled), stock, and a
user-supplied Roboflow project model. If the RF-DETR model fails to load,
the app falls back to the stock YOLOv8n detector.

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
detection first so the reported numbers are steady-state. The sweep also
prints a fifth section, `rfdetr_nano (Roboflow, bundled)`, for the default
RF-DETR detector.
