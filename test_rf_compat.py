"""
Compatibility tests for the Roboflow-shaped /infer/object_detection endpoint.

Validates against the REAL inference 1.7.3 classes:
  - a Roboflow-shaped request dict must parse as ObjectDetectionInferenceRequest
  - our JSON response must parse as ObjectDetectionInferenceResponse

Run:  ~/workspace/roboflow-spike/venv/bin/python test_rf_compat.py
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
)
from inference.core.entities.responses.inference import (  # noqa: E402
    ObjectDetectionInferenceResponse,
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

print(f"\nALL {len(passed)} CHECKS PASSED")
