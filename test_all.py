#!/usr/bin/env python3
"""Hit every inference endpoint once and print a short summary.

Usage:
    python3 test_all.py <image-file> [server-url]

Stdlib only, so it runs on the phone's system python3.
"""
import base64
import json
import sys
import time
import urllib.request

image_path = sys.argv[1] if len(sys.argv) > 1 else None
base_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:9001"

if not image_path:
    print(__doc__)
    sys.exit(1)

with open(image_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()


def post(path, body):
    req = urllib.request.Request(
        base_url + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())
    ms = (time.monotonic() - t0) * 1000
    return data, ms


img = {"image": {"type": "base64", "value": img_b64}}

print("== object_detection ==")
d, ms = post("/infer/object_detection", dict(img, max_detections=5))
ps = d["predictions"]
print(f"  {len(ps)} predictions, {ms:.1f} ms")
for p in ps[:3]:
    print(f"    {p['class']:<10} {p['confidence']:.3f}")

print("== instance_segmentation ==")
d, ms = post("/infer/instance_segmentation", dict(img, max_detections=3))
ps = d["predictions"]
print(f"  {len(ps)} predictions, {ms:.1f} ms")
if ps:
    print(f"    first mask: {len(ps[0]['points'])} polygon points, label={ps[0]['class']}")

print("== keypoint_detection ==")
d, ms = post("/infer/keypoint_detection", dict(img, max_detections=3))
ps = d["predictions"]
print(f"  {len(ps)} predictions, {ms:.1f} ms")
if ps:
    print(f"    first person: {len(ps[0]['keypoints'])} keypoints, label={ps[0]['class']}")

print("== classification ==")
d, ms = post("/infer/classification", img)
print(f"  top={d['top']} conf={d['confidence']:.3f}, {ms:.1f} ms")

print("done.")
