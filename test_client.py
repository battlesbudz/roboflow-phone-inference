#!/usr/bin/env python3
"""Smoke-test client for the lean inference server.

Usage:
    python test_client.py <image-file> [server-url]

Sends the image to POST /detect and prints the detections.
"""
import base64
import json
import sys
import urllib.request

image_path = sys.argv[1] if len(sys.argv) > 1 else None
base_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:9001"

if not image_path:
    print(__doc__)
    sys.exit(1)

with open(image_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

body = json.dumps({"image": img_b64}).encode()
req = urllib.request.Request(
    base_url + "/detect", data=body, headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req, timeout=60) as r:
    out = json.loads(r.read())

print(f"inference_ms: {out['inference_ms']}")
for d in out["detections"]:
    b = d["box"]
    print(f"  {d['label']:<12} {d['confidence']:.3f}  ({b['x1']}, {b['y1']})-({b['x2']}, {b['y2']})")
if not out["detections"]:
    print("  no detections above threshold")
