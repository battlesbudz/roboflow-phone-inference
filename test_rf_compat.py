"""
Compatibility tests for the Roboflow-shaped /infer/* endpoints:

  - /infer/object_detection
  - /infer/instance_segmentation
  - /infer/keypoint_detection
  - /infer/classification

Validates against the REAL inference 1.7.3 classes:
  - a Roboflow-shaped request dict must parse as the matching
    *InferenceRequest class
  - our JSON response must parse as the matching *InferenceResponse
    class (or its dataclass twin where the real one is a dataclass)

Run from this directory:
  ~/workspace/roboflow-spike/venv/bin/python test_rf_compat.py
"""

import base64
import io
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(__file__).rsplit("/", 1)[0])
import server  # noqa: E402
from server import app, Detection, Box  # noqa: E402

# Real Roboflow classes (inference 1.7.3). Slow import, done once.
from inference.core.entities.requests.inference import (  # noqa: E402
    ObjectDetectionInferenceRequest,
    InstanceSegmentationInferenceRequest,
    KeypointsDetectionInferenceRequest,
    ClassificationInferenceRequest,
)
from inference.core.entities.responses.inference import (  # noqa: E402
    ObjectDetectionInferenceResponse,
    InstanceSegmentationPredictionDC,
    PointDC,
    KeypointsDetectionInferenceResponse,
    ClassificationInferenceResponse,
)

client = TestClient(app)
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

print("== /health ==")
r = client.get("/health")
check("health 200", r.status_code == 200)
check("health status ok", r.json()["status"] == "ok")

print("== /detect (unchanged) ==")
r = client.post("/detect", json={"image": B64})
check("detect 200", r.status_code == 200)
check("detect has detections+inference_ms",
      "detections" in r.json() and "inference_ms" in r.json())
r = client.post("/detect", json={"image": "!!!not-base64!!!"})
check("detect bad base64 -> 400", r.status_code == 400)

print("== request shape vs real ObjectDetectionInferenceRequest ==")
rf_request = {
    "image": {"type": "base64", "value": B64},
    "api_key": "ignored-locally",
    "model_id": "ignored-locally",
    "confidence": 0.5,
    "iou_threshold": 0.3,
    "class_filter": ["person", "car"],
    "max_detections": 10,
}
parsed = ObjectDetectionInferenceRequest(**rf_request)
check("real request class parses our shape", parsed.confidence == 0.5)

print("== POST /infer/object_detection ==")
r = client.post("/infer/object_detection", json=rf_request)
check("infer 200", r.status_code == 200)
body = r.json()

print("== response shape vs real ObjectDetectionInferenceResponse ==")
resp = ObjectDetectionInferenceResponse(**body)
check("real response class parses our output", isinstance(resp, ObjectDetectionInferenceResponse))
check("image dims echoed", body["image"] == {"width": 320, "height": 240})
check("has inference_id", isinstance(body.get("inference_id"), str))
check("has time", isinstance(body.get("time"), (int, float)))

print("== conversion math (stubbed detector, deterministic) ==")
stub_dets = [
    Detection(label="person", confidence=0.9, box=Box(x1=100.0, y1=50.0, x2=200.0, y2=150.0)),
    Detection(label="car", confidence=0.7, box=Box(x1=0.0, y1=0.0, x2=40.0, y2=30.0)),
]


class StubDetector:
    def detect(self, image, confidence=0.25, iou_threshold=0.45,
               class_filter=None, max_detections=None):
        dets = [d for d in stub_dets if d.confidence >= confidence]
        if class_filter is not None:
            dets = [d for d in dets if d.label in class_filter]
        if max_detections is not None:
            dets = dets[:max_detections]
        return dets, 1.0


real_detector = server.detector
server.detector = StubDetector()
try:
    r = client.post("/infer/object_detection", json={
        "image": {"type": "base64", "value": B64}, "confidence": 0.5})
    preds = r.json()["predictions"]
    check("two predictions", len(preds) == 2)
    p = preds[0]
    check("center x", p["x"] == 150.0)
    check("center y", p["y"] == 100.0)
    check("width", p["width"] == 100.0)
    check("height", p["height"] == 100.0)
    check("class alias", p["class"] == "person")
    check("class_id", p["class_id"] == 0)  # person is COCO id 0
    check("confidence", p["confidence"] == 0.9)
    check("detection_id present", isinstance(p["detection_id"], str) and len(p["detection_id"]) > 0)
    # real class still parses the stubbed output
    ObjectDetectionInferenceResponse(**r.json())
    check("real response class parses stubbed output", True)

    r = client.post("/infer/object_detection", json={
        "image": {"type": "base64", "value": B64},
        "class_filter": ["car"]})
    preds = r.json()["predictions"]
    check("class_filter applied", len(preds) == 1 and preds[0]["class"] == "car")

    r = client.post("/infer/object_detection", json={
        "image": {"type": "base64", "value": B64}, "max_detections": 1})
    check("max_detections applied", len(r.json()["predictions"]) == 1)

    r = client.post("/infer/object_detection", json={
        "image": {"type": "base64", "value": B64}, "confidence": "default"})
    check('confidence "default" accepted', r.status_code == 200)

    r = client.post("/infer/object_detection", json={
        "image": {"type": "base64", "value": B64}, "confidence": "nonsense"})
    check('confidence "nonsense" -> 400', r.status_code == 400)

    r = client.post("/infer/object_detection", json={
        "image": {"type": "base64", "value": B64}, "confidence": 1.5})
    check("confidence 1.5 -> 400", r.status_code == 400)
finally:
    server.detector = real_detector

print("== malformed input ==")
r = client.post("/infer/object_detection", json={
    "image": {"type": "numpy", "value": "x"}})
check('image type "numpy" -> 400', r.status_code == 400)
r = client.post("/infer/object_detection", json={
    "image": {"type": "base64", "value": "!!!not-base64!!!"}})
check("bad base64 -> 400", r.status_code == 400)
r = client.post("/infer/object_detection", json={
    "image": {"type": "base64", "value": base64.b64encode(b"not an image").decode()}})
check("undecodable bytes -> 400", r.status_code == 400)

print("== image type url (local http server) ==")
img_bytes = io.BytesIO()
Image.new("RGB", (160, 120), (10, 200, 90)).save(img_bytes, format="PNG")
PNG = img_bytes.getvalue()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(PNG)))
        self.end_headers()
        self.wfile.write(PNG)

    def log_message(self, *a):
        pass


httpd = HTTPServer(("127.0.0.1", 0), Handler)
port = httpd.server_address[1]
t = threading.Thread(target=httpd.serve_forever, daemon=True)
t.start()
r = client.post("/infer/object_detection", json={
    "image": {"type": "url", "value": f"http://127.0.0.1:{port}/t.png"}})
check("url image 200", r.status_code == 200)
check("url image dims", r.json()["image"] == {"width": 160, "height": 120})
httpd.shutdown()

r = client.post("/infer/object_detection", json={
    "image": {"type": "url", "value": "http://127.0.0.1:1/none.png"}})
check("unreachable url -> 400", r.status_code == 400)

print("== new request shapes vs real 1.7.3 classes ==")
BUS_B64 = base64.b64encode(open("test-bus.jpg", "rb").read()).decode()
BUS_W, BUS_H = 810, 1080
seg_req = {
    "image": {"type": "base64", "value": BUS_B64},
    "api_key": "ignored-locally",
    "model_id": "ignored-locally",
    "confidence": 0.5,
    "iou_threshold": 0.3,
    "class_filter": ["person", "bus"],
    "max_detections": 10,
    "mask_decode_mode": "accurate",
}
parsed = InstanceSegmentationInferenceRequest(**seg_req)
check("real seg request class parses our shape", parsed.mask_decode_mode == "accurate")
pose_req = dict(seg_req)
pose_req.update({"keypoint_confidence": 0.5, "keypoint_iou_threshold": 0.5})
del pose_req["mask_decode_mode"]
parsed = KeypointsDetectionInferenceRequest(**pose_req)
check("real pose request class parses our shape", parsed.keypoint_confidence == 0.5)
cls_req = {
    "image": {"type": "base64", "value": BUS_B64},
    "api_key": "ignored-locally",
    "model_id": "ignored-locally",
    "confidence": 0.5,
}
parsed = ClassificationInferenceRequest(**cls_req)
check("real classification request class parses our shape", parsed.confidence == 0.5)

print("== POST /infer/instance_segmentation (real model) ==")
r = client.post("/infer/instance_segmentation", json=seg_req)
check("seg 200", r.status_code == 200)
body = r.json()
check("seg has inference_id+time", isinstance(body.get("inference_id"), str)
      and isinstance(body.get("time"), (int, float)))
check("seg image dims echoed", body["image"] == {"width": BUS_W, "height": BUS_H})
check("seg has predictions", isinstance(body["predictions"], list)
      and len(body["predictions"]) > 0)
p0 = body["predictions"][0]
for k in ("x", "y", "width", "height", "confidence", "class", "class_id",
          "detection_id", "points"):
    check(f"seg pred has {k}", k in p0)
check("seg coords in image",
      0 <= p0["x"] <= BUS_W and 0 <= p0["y"] <= BUS_H
      and 0 < p0["width"] <= BUS_W and 0 < p0["height"] <= BUS_H)
check("seg confidence range", 0.0 <= p0["confidence"] <= 1.0)
check("seg class sane", p0["class"] in server.COCO_LABELS
      and p0["class_id"] == server.COCO_LABELS.index(p0["class"]))
check("seg detection_id present", isinstance(p0["detection_id"], str)
      and len(p0["detection_id"]) > 0)
check("seg points non-empty", isinstance(p0["points"], list) and len(p0["points"]) >= 3)
pt0 = p0["points"][0]
check("seg point keys", set(pt0.keys()) == {"x", "y"})
check("seg points in image",
      all(0 <= q["x"] <= BUS_W and 0 <= q["y"] <= BUS_H for q in p0["points"]))
# the real 1.7.3 dataclass twin accepts our first prediction's shape
InstanceSegmentationPredictionDC(
    x=p0["x"], y=p0["y"], width=p0["width"], height=p0["height"],
    confidence=p0["confidence"], class_name=p0["class"], class_id=p0["class_id"],
    points=[PointDC(x=q["x"], y=q["y"]) for q in p0["points"]],
    detection_id=p0["detection_id"],
)
check("real 1.7.3 seg dataclass accepts our shape", True)

r = client.post("/infer/instance_segmentation", json={
    "image": {"type": "base64", "value": BUS_B64}, "max_detections": 2})
check("seg max_detections applied", len(r.json()["predictions"]) <= 2)
r = client.post("/infer/instance_segmentation", json={
    "image": {"type": "base64", "value": "!!!not-base64!!!"}})
check("seg bad base64 -> 400", r.status_code == 400)
r = client.post("/infer/instance_segmentation", json={
    "image": {"type": "base64", "value": BUS_B64}, "confidence": "nonsense"})
check('seg confidence "nonsense" -> 400', r.status_code == 400)

print("== POST /infer/keypoint_detection (real model) ==")
r = client.post("/infer/keypoint_detection", json={
    **pose_req, "keypoint_confidence": 0.0})
check("pose 200", r.status_code == 200)
body = r.json()
check("pose has inference_id+time", isinstance(body.get("inference_id"), str)
      and isinstance(body.get("time"), (int, float)))
check("pose image dims echoed", body["image"] == {"width": BUS_W, "height": BUS_H})
check("pose has predictions", isinstance(body["predictions"], list)
      and len(body["predictions"]) > 0)
p0 = body["predictions"][0]
for k in ("x", "y", "width", "height", "confidence", "class", "class_id",
          "detection_id", "keypoints"):
    check(f"pose pred has {k}", k in p0)
check("pose is person", p0["class"] == "person" and p0["class_id"] == 0)
check("pose box in image",
      0 <= p0["x"] <= BUS_W and 0 <= p0["y"] <= BUS_H
      and 0 < p0["width"] <= BUS_W and 0 < p0["height"] <= BUS_H)
check("pose has 17 keypoints", isinstance(p0["keypoints"], list)
      and len(p0["keypoints"]) == 17)
k0 = p0["keypoints"][0]
check("pose keypoint keys", set(k0.keys()) == {"x", "y", "confidence", "class", "class_id"})
check("pose keypoint names", [k["class"] for k in p0["keypoints"]]
      == server.COCO_KEYPOINT_NAMES)
check("pose keypoint ids", [k["class_id"] for k in p0["keypoints"]] == list(range(17)))
check("pose keypoints in image",
      all(0 <= k["x"] <= BUS_W and 0 <= k["y"] <= BUS_H for k in p0["keypoints"]))
check("pose keypoint confidences",
      all(0.0 <= k["confidence"] <= 1.0 for k in p0["keypoints"]))
KeypointsDetectionInferenceResponse(**body)
check("real 1.7.3 pose response class parses our output", True)

r = client.post("/infer/keypoint_detection", json={
    "image": {"type": "base64", "value": BUS_B64}, "keypoint_confidence": 0.99})
n_loose = len(client.post("/infer/keypoint_detection", json={
    "image": {"type": "base64", "value": BUS_B64},
    "keypoint_confidence": 0.0}).json()["predictions"][0]["keypoints"])
n_strict = len(r.json()["predictions"][0]["keypoints"]) if r.json()["predictions"] else 0
check("keypoint_confidence filters", n_strict <= n_loose)
r = client.post("/infer/keypoint_detection", json={
    "image": {"type": "base64", "value": BUS_B64}, "confidence": "best"})
check('pose confidence "best" -> 400 (parity with 1.7.3)', r.status_code == 400)
r = client.post("/infer/keypoint_detection", json={
    "image": {"type": "base64", "value": "!!!not-base64!!!"}})
check("pose bad base64 -> 400", r.status_code == 400)

print("== POST /infer/classification (real model) ==")
r = client.post("/infer/classification", json=cls_req)
check("cls 200", r.status_code == 200)
body = r.json()
for k in ("top", "confidence", "predictions", "image", "inference_id", "time"):
    check(f"cls has {k}", k in body)
check("cls top non-empty", isinstance(body["top"], str) and len(body["top"]) > 0)
check("cls confidence range", 0.0 <= body["confidence"] <= 1.0)
check("cls top matches", body["top"] == body["predictions"][0]["class"]
      and body["confidence"] == body["predictions"][0]["confidence"])
check("cls top-5", len(body["predictions"]) == 5)
check("cls sorted desc",
      all(body["predictions"][i]["confidence"] >= body["predictions"][i + 1]["confidence"]
          for i in range(4)))
check("cls class_ids sane",
      all(isinstance(p["class_id"], int) and 0 <= p["class_id"] < 1000
          for p in body["predictions"]))
check("cls image dims echoed", body["image"] == {"width": BUS_W, "height": BUS_H})
check("cls has inference_id+time", isinstance(body.get("inference_id"), str)
      and isinstance(body.get("time"), (int, float)))
ClassificationInferenceResponse(**body)
check("real 1.7.3 classification response class parses our output", True)

r = client.post("/infer/classification", json={
    "image": {"type": "base64", "value": BUS_B64}, "confidence": 0.9999})
b2 = r.json()
check("cls high threshold clears top", b2["top"] == "" and b2["confidence"] == 0.0)
check("cls high threshold keeps top-5", len(b2["predictions"]) == 5)
r = client.post("/infer/classification", json={
    "image": {"type": "base64", "value": "!!!not-base64!!!"}})
check("cls bad base64 -> 400", r.status_code == 400)

print(f"\nALL {len(passed)} CHECKS PASSED")
