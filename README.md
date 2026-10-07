# Phone Inference

A tiny, self-contained object-detection server that runs on a phone.

ONNX Runtime + YOLOv8n behind a two-endpoint HTTP API. The full
Roboflow inference server does this job in 7.3 GB of dependencies and
~1.1 GB of idle RAM; this does it in ~130 MB on disk and ~160 MB of RAM,
which is the difference between "runs in a datacenter" and "runs in
Termux on your phone."

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

Full contract: [API.md](API.md).

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

- `server.py` — the whole server (~200 lines)
- `termux-setup.sh` — one-shot phone bootstrap
- `yolov8n.onnx` — YOLOv8 nano, COCO 80 classes (12 MB)
- `test_client.py` — smoke-test client
- `API.md` / `BENCHMARKS.md` / `TERMUX.md` — docs

## Roadmap

- On-device benchmark numbers (in progress)
- Native Android (Kotlin + ONNX Runtime, NNAPI delegate) version to
  replace the Termux/Python layer
- Direct integration with the on-device assistant's tool loop

## License

MIT. See [LICENSE](LICENSE).
