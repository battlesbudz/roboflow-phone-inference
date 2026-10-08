# Running on a phone (Termux)

Tested path: Termux on Android (aarch64). The one-shot script handles the
fiddly parts; this guide explains each step and what to do when something
complains.

## Prerequisites

- Termux from **F-Droid** (https://f-droid.org/packages/com.termux/).
  The Play Store build is years out of date and will not work.
- About 500 MB free (Python + packages + model).
- A network connection for the one-time install.

## Quick path

```bash
mkdir -p ~/jarvis-vision && cd ~/jarvis-vision
# copy server.py, termux-setup.sh, the four .onnx files, and synset.txt
# into this directory
bash termux-setup.sh
```

What the script does:

1. `pkg update && pkg install python python-numpy libjpeg-turbo`
   (numpy comes from Termux's own repo as a prebuilt binary; installing it
   via pip on Android triggers long source builds that often fail)
2. Creates a venv with `--system-site-packages` so the venv sees that numpy
3. `pip install pillow fastapi "uvicorn[standard]" pydantic`
4. `pip install onnxruntime` (see "If onnxruntime won't install" below)
5. Verifies the import, checks for the four `.onnx` model files and
   `synset.txt`, writes `run.sh`

Then:

```bash
bash ~/jarvis-vision/run.sh
```

In a second Termux session (swipe from the left edge, "New session"):

```bash
curl http://localhost:9001/health
```

Expected: `{"status":"ok","model":"yolov8n",...}`.

## Sending a test image

From your computer, or from a third Termux session with the server running:

```bash
python test_client.py /path/to/photo.jpg http://localhost:9001
```

To test with a photo taken on the phone: Termux exposes the camera via
`termux-camera-photo` (needs `pkg install termux-api` plus the Termux:API
app from F-Droid):

```bash
pkg install termux-api
termux-camera-photo -c 0 /sdcard/frame.jpg
python test_client.py /sdcard/frame.jpg http://localhost:9001
```

## If onnxruntime won't install

`pip install onnxruntime` works on most recent Termux + aarch64 setups, but
PyPI does not publish wheels for every Android ABI. If pip fails:

1. Try the Termux system package: `pkg install onnxruntime`
   (the setup script already tries this automatically).
2. Community-built Termux wheels exist (e.g. the `wheel-forge` project on
   GitHub ships `python-onnxruntime` `.deb`s for aarch64); install with
   `dpkg -i` and re-run the verify step.
3. Last resort: `onnxruntime-mobile` via a manual wheel. The server only
   uses the basic `InferenceSession` API, so any reasonably recent build
   works.

If you get it working on a device where the script failed, please open an
issue noting the device, Android version, and what fixed it.

## Keeping it running

- Termux sessions die when Android reclaims them. For casual testing this is
  fine; for anything long-lived, look at `termux-wake-lock` and turning off
  battery optimization for the Termux app.
- The server binds to `127.0.0.1` only. Nothing leaves the phone.

## Known limitations

- CPU only. ONNX Runtime's Android NNAPI/GPU delegates are not wired up
  here; a future native (Kotlin) version of this server will use them.
- Four models load at startup (object detection, segmentation, pose,
  classification — about 55 MB of weights total, plus runtime memory).
