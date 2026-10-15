#!/usr/bin/env python3
"""Lightweight smoke test for the /infer/* endpoints — no `inference` dep.

Unlike test_rf_compat.py (which validates shapes against the real Roboflow
inference 1.7.3 classes and needs the 7 GB inference package), this file
uses only the server's own requirements and is what CI runs on every push:
  - /health, /detect, and all four /infer/* endpoints return 200
  - response shapes carry every required key
  - image dims are echoed, inference_id/time present
  - real-model positive checks on test-bus.jpg (seg + pose find something)
  - malformed input is rejected with 400

Run from this directory:
    python test_smoke.py
"""
import base64
import io
import sys

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(__file__).rsplit("/", 1)[0])
import server  # noqa: E402

client = TestClient(server.app)
passed = []


def check(name, cond):
    assert cond, f"FAILED: {name}"
    passed.append(name)
    print(f"  ok: {name}")


def make_image_b64(w=320, h=240, color=(200, 100, 50)):
    img = Image.new("RGB", (w, h), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


B64 = make_image_b64()
IMG = {"type": "base64", "value": B64}

print("== /health ==")
r = client.get("/health")
check("health 200", r.status_code == 200)
check("health status ok", r.json()["status"] == "ok")
check("health lists all four models", set(r.json()["models"]) == {
    "object_detection", "instance_segmentation",
    "keypoint_detection", "classification"})

print("== /detect ==")
r = client.post("/detect", json={"image": B64})
check("detect 200", r.status_code == 200)
check("detect shape", "detections" in r.json() and "inference_ms" in r.json())
r = client.post("/detect", json={"image": "!!!not-base64!!!"})
check("detect bad base64 -> 400", r.status_code == 400)

print("== /infer/object_detection ==")
r = client.post("/infer/object_detection", json={
    "image": IMG, "confidence": 0.5, "iou_threshold": 0.3,
    "class_filter": ["person", "car"], "max_detections": 10,
    "api_key": "ignored-locally", "model_id": "ignored-locally"})
check("od 200", r.status_code == 200)
body = r.json()
check("od image dims echoed", body["image"] == {"width": 320, "height": 240})
check("od has inference_id+time",
      isinstance(body.get("inference_id"), str)
      and isinstance(body.get("time"), (int, float)))
for p in body["predictions"]:
    check("od pred keys",
          {"x", "y", "width", "height", "confidence", "class",
           "class_id", "detection_id"} <= set(p))
    check("od pred in image",
          0 <= p["x"] <= 320 and 0 <= p["y"] <= 240
          and 0 < p["width"] <= 320 and 0 < p["height"] <= 240)
    check("od confidence range", 0.0 <= p["confidence"] <= 1.0)

print("== malformed input ==")
r = client.post("/infer/object_detection",
                json={"image": {"type": "numpy", "value": "x"}})
check('image type "numpy" -> 400', r.status_code == 400)
r = client.post("/infer/object_detection",
                json={"image": {"type": "base64", "value": "!!!not-base64!!!"}})
check("bad base64 -> 400", r.status_code == 400)
r = client.post("/infer/object_detection",
                json={"image": IMG, "confidence": "nonsense"})
check('confidence "nonsense" -> 400', r.status_code == 400)
r = client.post("/infer/object_detection",
                json={"image": IMG, "confidence": 1.5})
check("confidence 1.5 -> 400", r.status_code == 400)
r = client.post("/infer/keypoint_detection",
                json={"image": IMG, "confidence": "best"})
check('pose confidence "best" -> 400', r.status_code == 400)

print("== real-model checks on test-bus.jpg ==")
try:
    BUS_B64 = base64.b64encode(open("test-bus.jpg", "rb").read()).decode()
except FileNotFoundError:
    print("  SKIP: test-bus.jpg not present (real-model checks need it)")
    BUS_B64 = None

if BUS_B64:
    BUS = {"type": "base64", "value": BUS_B64}
    BUS_W, BUS_H = Image.open(io.BytesIO(
        base64.b64decode(BUS_B64))).size
    r = client.post("/infer/instance_segmentation", json={
        "image": BUS, "confidence": 0.5, "max_detections": 10,
        "mask_decode_mode": "accurate"})
    check("seg 200", r.status_code == 200)
    body = r.json()
    check("seg image dims echoed", body["image"] == {"width": BUS_W, "height": BUS_H})
    check("seg finds something", len(body["predictions"]) > 0)
    p0 = body["predictions"][0]
    check("seg pred keys",
          {"x", "y", "width", "height", "confidence", "class",
           "class_id", "detection_id", "points"} <= set(p0))
    check("seg points non-empty",
          isinstance(p0["points"], list) and len(p0["points"]) >= 3)
    check("seg points in image",
          all(0 <= q["x"] <= BUS_W and 0 <= q["y"] <= BUS_H
              for q in p0["points"]))
    check("seg point keys",
          all(set(q.keys()) == {"x", "y"} for q in p0["points"]))

    r = client.post("/infer/keypoint_detection", json={
        "image": BUS, "confidence": 0.5, "keypoint_confidence": 0.0})
    check("pose 200", r.status_code == 200)
    body = r.json()
    check("pose image dims echoed", body["image"] == {"width": BUS_W, "height": BUS_H})
    check("pose finds a person", len(body["predictions"]) > 0)
    p0 = body["predictions"][0]
    check("pose pred keys",
          {"x", "y", "width", "height", "confidence", "class",
           "class_id", "detection_id", "keypoints"} <= set(p0))
    check("pose is person", p0["class"] == "person" and p0["class_id"] == 0)
    check("pose has 17 keypoints", len(p0["keypoints"]) == 17)
    check("pose keypoint names",
          [k["class"] for k in p0["keypoints"]] == server.COCO_KEYPOINT_NAMES)
    check("pose keypoint ids",
          [k["class_id"] for k in p0["keypoints"]] == list(range(17)))
    check("pose keypoints in image",
          all(0 <= k["x"] <= BUS_W and 0 <= k["y"] <= BUS_H
              for k in p0["keypoints"]))

    r = client.post("/infer/classification", json={"image": BUS, "confidence": 0.5})
    check("cls 200", r.status_code == 200)
    body = r.json()
    for k in ("top", "confidence", "predictions", "image",
              "inference_id", "time"):
        check(f"cls has {k}", k in body)
    check("cls top-5", len(body["predictions"]) == 5)
    check("cls sorted desc",
          all(body["predictions"][i]["confidence"]
              >= body["predictions"][i + 1]["confidence"] for i in range(4)))
    check("cls top matches",
          body["top"] == body["predictions"][0]["class"]
          and body["confidence"] == body["predictions"][0]["confidence"])
    check("cls image dims echoed", body["image"] == {"width": BUS_W, "height": BUS_H})
    r = client.post("/infer/classification",
                    json={"image": BUS, "confidence": 0.9999})
    b2 = r.json()
    check("cls high threshold clears top",
          b2["top"] == "" and b2["confidence"] == 0.0)
    check("cls high threshold keeps top-5", len(b2["predictions"]) == 5)

print(f"\nALL {len(passed)} CHECKS PASSED")
