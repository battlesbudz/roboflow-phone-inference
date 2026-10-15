"""Verify an RF-DETR ONNX model against Roboflow's own decode contract.

Runs the model on an image with onnxruntime, decodes the raw
``dets`` / ``labels`` outputs exactly the way Roboflow's
``rfdetr.export._runtime.decode.decode_detections`` does, and reports
detections + latency. Used by:
  - ``export_rfdetr.py`` (export-time check: fp32 vs fp16 parity)
  - ``.github/workflows/verify-rfdetr-model.yml`` (every push touching the
    bundled model or its Kotlin decoder)

Decode contract (mirrors Roboflow's helpers):
  - Preprocess: direct resize to the model's input size (no letterbox),
    /255, ImageNet mean/std, NCHW float32.
  - Outputs: ``dets`` = (1, Q, 4) normalised cxcywh, ``labels`` =
    (1, Q, C) raw logits.
  - Per-class sigmoid (logits clipped to [-88, 88]), keep every class
    slot (official COCO weights: the final slot is the real class 90,
    not background), take the top-Q query/class pairs, keep those with
    score > threshold. No NMS: DETR queries do not duplicate boxes.
  - Slot index = COCO category ID; the sparse map below is Roboflow's
    ``COCO_CLASSES`` (rfdetr/assets/coco_classes.py). Slots with no
    category (0, 12, 26, ...) never emit a detection.
"""

import argparse
import json
import sys
import time

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Sparse COCO category ID -> name. Verbatim from Roboflow's
# rfdetr/assets/coco_classes.py (COCO_CLASSES). Keys are COCO category
# IDs (1-90, with gaps); there is no category 0.
COCO_CATEGORY_TO_NAME = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie",
    33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard",
    37: "sports ball", 38: "kite", 39: "baseball bat", 40: "baseball glove",
    41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle",
    46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon",
    51: "bowl", 52: "banana", 53: "apple", 54: "sandwich", 55: "orange",
    56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut",
    61: "cake", 62: "chair", 63: "couch", 64: "potted plant", 65: "bed",
    67: "dining table", 70: "toilet", 72: "tv", 73: "laptop", 74: "mouse",
    75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave",
    79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book",
    85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear",
    89: "hair drier", 90: "toothbrush",
}


def preprocess(image_path, height, width):
    """PIL resize + /255 + ImageNet normalize -> (1, 3, H, W) float32."""
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    resized = img.resize((width, height), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    chw = arr.transpose(2, 0, 1)[np.newaxis, :, :, :]
    return chw.astype(np.float32), orig_w, orig_h


def decode(dets, labels, orig_w, orig_h, threshold=0.3):
    """dets: (Q, 4) normalised cxcywh. labels: (Q, C) raw logits.

    Returns a list of dicts sorted by confidence desc.
    """
    q, c = labels.shape
    logits = np.clip(labels, -88.0, 88.0)
    scores_all = 1.0 / (1.0 + np.exp(-logits))
    # Flatten (Q, C) -> Q*C pairs; take top-Q before thresholding,
    # mirroring PostProcess._select_topk / _select_topk_multiclass.
    flat = scores_all.reshape(-1)
    order = np.argsort(-flat, kind="stable")[:q]
    out = []
    for idx in order:
        score = float(flat[idx])
        if score <= threshold:
            break  # order is descending, so nothing later clears it
        query = int(idx // c)
        slot = int(idx % c)
        name = COCO_CATEGORY_TO_NAME.get(slot)
        if name is None:
            continue  # unmapped slot (0, 12, 26, ...): no COCO category
        cx, cy, bw, bh = [float(v) for v in dets[query]]
        x1 = max(0.0, (cx - bw / 2) * orig_w)
        y1 = max(0.0, (cy - bh / 2) * orig_h)
        x2 = min(float(orig_w), (cx + bw / 2) * orig_w)
        y2 = min(float(orig_h), (cy + bh / 2) * orig_h)
        out.append({
            "label": name,
            "class_id": slot,  # COCO category ID (sparse), not a dense index
            "confidence": round(score, 3),
            "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        })
    return out


def run_model(model_path, image_path, threshold=0.3, iters=3):
    """Returns a dict with input/output shapes, detections, latency."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    _, _, h, w = [int(d) for d in inp.shape]
    tensor, orig_w, orig_h = preprocess(image_path, h, w)
    out_names = [o.name for o in sess.get_outputs()]
    dets_idx = next((i for i, n in enumerate(out_names) if "dets" in n), 0)
    labels_idx = next((i for i, n in enumerate(out_names) if "labels" in n), 1)
    # Warmup (kernel init), then timed runs.
    sess.run(None, {inp.name: tensor})
    lat = []
    raw = None
    for _ in range(iters):
        t0 = time.perf_counter()
        raw = sess.run(None, {inp.name: tensor})
        lat.append((time.perf_counter() - t0) * 1000.0)
    dets = np.asarray(raw[dets_idx])[0]
    labels = np.asarray(raw[labels_idx])[0]
    detections = decode(dets, labels, orig_w, orig_h, threshold)
    return {
        "input": {"name": inp.name, "shape": [int(d) for d in inp.shape]},
        "outputs": [
            {"name": n, "shape": [int(d) for d in o.shape]}
            for n, o in zip(out_names, sess.get_outputs())
        ],
        "detections": detections,
        "latency_ms": {
            "mean": round(sum(lat) / len(lat), 1),
            "runs": [round(v, 1) for v in lat],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--require-label", action="append", default=[],
                    help="label that must appear among detections (repeatable)")
    ap.add_argument("--min-detections", type=int, default=1)
    ap.add_argument("--json", default=None, help="write full result JSON here")
    args = ap.parse_args()

    result = run_model(args.model, args.image, args.threshold)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)

    labels = [d["label"] for d in result["detections"]]
    print(f"model: {args.model}")
    print(f"input: {result['input']['name']} {result['input']['shape']}")
    for o in result["outputs"]:
        print(f"output: {o['name']} {o['shape']}")
    print(f"latency_ms mean: {result['latency_ms']['mean']} runs: {result['latency_ms']['runs']}")
    print(f"detections ({len(labels)}):")
    for d in result["detections"][:10]:
        print(f"  {d['label']:15s} {d['confidence']:.3f} {d['box']}")

    ok = True
    if len(labels) < args.min_detections:
        print(f"FAIL: expected >= {args.min_detections} detections, got {len(labels)}")
        ok = False
    for want in args.require_label:
        if want not in labels:
            print(f"FAIL: expected label '{want}' among detections, got {labels[:10]}")
            ok = False
    if not ok:
        sys.exit(1)
    print("VERIFY OK")


if __name__ == "__main__":
    main()
