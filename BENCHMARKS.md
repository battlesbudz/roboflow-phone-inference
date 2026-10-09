# Benchmarks

Measured 2026-10-06. All numbers are from the same weak test machine unless
noted: 2 shared vCPUs (AMD EPYC 9D64), no GPU. A modern phone SoC should beat
the per-frame numbers here; phone results were measured 2026-10-08 (see "On-device results" below).

## Headline: full inference server vs lean server

| Metric | Full Roboflow inference server (`inference-cpu[http]`) | Lean server (this repo) |
|---|---|---|
| Installed size | 7.3 GB (venv) | ~120 MB Python packages + 12 MB model |
| RAM at idle (RSS) | ~1.1 GB | ~160 MB |
| Cold start to serving | 38.4 s | ~2 s |
| Model load (YOLOv8n ONNX) | ~0.1 s | ~0.16 s |
| Per-frame, 640x640 | 259–351 ms (direct ONNX Runtime calls) | ~117 ms server-side (200 ms HTTP round-trip) |

The per-frame gap between the two columns is mostly measurement setup, not
architecture: the "full server" number is raw ONNX Runtime with no HTTP
overhead, while the lean number is end-to-end through the HTTP API. The
point of the table is everything above that row: the full server carries
roughly 60x the disk and 7x the idle RAM for the same model and the same
inference engine.

## How each number was measured

- **Installed size**: `du -sh` on the virtualenv after `pip install`.
- **RAM at idle**: `ps -o rss` on the server process after startup, before
  any inference request.
- **Cold start**: wall time from launching the server process to the first
  successful `GET /health` (full server: first `GET /docs` 200).
- **Model load**: wall time around `onnxruntime.InferenceSession(...)`.
- **Per-frame (full server)**: 4 timed `session.run()` calls on a 640x640
  RGB input: 351, 329, 259, 273 ms.
- **Per-frame (lean)**: `inference_ms` field returned by `POST /detect`
  (server-side model time, excludes HTTP/base64/decode). Round-trip measured
  separately with `time.time()` around the HTTP call.

## PyTorch vs ONNX, same weights

Same YOLOv8n weights, same machine, 640x640 input:

| | PyTorch (`yolov8n.pt`) | ONNX Runtime (`yolov8n.onnx`) |
|---|---|---|
| Model load | 10.8 s | ~0.1–0.16 s |
| First prediction | ~9.1 s | ~0.3 s |
| Steady-state frame | not measured (too slow to matter) | 259–351 ms |

On a CPU-only device the PyTorch path is not viable for anything
interactive. ONNX Runtime is the whole game, which is why this repo is
built on it directly.

## On-device results

Pending. The Termux install path (`TERMUX.md`) is designed so the same
`POST /detect` contract can be timed on a real phone. A public project
running YOLO + ONNX Runtime on Termux reports 12 FPS on a Samsung Galaxy
A50 (2019 mid-range), which suggests a 2024 flagship will be comfortably
faster than the numbers above.

All numbers below are HTTP round-trip via `test_all.py` on the same
`test-bus.jpg` image, same `/infer/*` endpoints, 2026-10-08 (phone) and
2026-10-09 (VM). Round-trip includes base64 encode, HTTP, server inference,
and JSON decode.

| Endpoint | Sandbox VM (2x shared AMD EPYC vCPU, no GPU) | Galaxy Z Fold 6 (Snapdragon 8 Gen 3, Termux) |
|---|---|---|
| Object detection (YOLOv8n, 5 preds) | 404.3 ms | 285.3 ms |
| Instance segmentation (YOLOv8n-seg, 3 preds, 134-pt mask) | 367.9 ms | 238.8 ms |
| Keypoint detection (YOLOv8n-pose, 3 people x 17 kpts) | 378.3 ms | 137.2 ms |
| Classification (MobileNetV2, minibus 0.723) | 43.0 ms | 50.1 ms |

The phone beats the weak shared VM vCPUs on every endpoint except
classification (near tie). Model load on the VM: detection 201 ms,
segmentation 241 ms, pose 143 ms, classification 211 ms.

## Reproducing

```bash
pip install -r requirements.txt
python -m uvicorn server:app --host 127.0.0.1 --port 9001
# in another shell:
python test_client.py <any jpg/png>
```

`test_client.py` prints the server-reported `inference_ms` per frame.
