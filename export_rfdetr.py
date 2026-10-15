"""One-off export: RF-DETR Nano (Roboflow) -> ONNX, dynamically quantized to int8,
with verification.

Run in CI via .github/workflows/export-rfdetr.yml (workflow_dispatch).
It must NOT be run to produce production artifacts from anywhere else:
the committed model file's provenance is exactly this script + the
export workflow + the provenance JSON it writes.

Pipeline:
  1. Instantiate Roboflow's RFDETRNano() (downloads the official COCO
     pretrained checkpoint from HuggingFace, roboflow org) and export
     to ONNX with Roboflow's own rfdetr export code (opset 17, fp32).
  2. Dynamically quantize with ONNX Runtime's own
     ``onnxruntime.quantization.quantize_dynamic`` (MatMul / Gemm /
     Attention weights to int8; Conv layers stay fp32, which is all
     dynamic quantization supports). ORT produces the quantized graph,
     so ORT can load it by construction.
  3. Verify fp32 vs quantized on test-bus.jpg with verify_rfdetr.py's
     decode (Roboflow's contract): same SET of top-5 labels, scores
     within 0.1. Quantization is lossy, so the tolerance is looser
     than exact parity -- but a garbage quantization still fails.
  4. Write RFDETR_PROVENANCE.json and copy the quantized ONNX to
     rf-detr-nano.onnx at the repo root. The workflow commits both.

Why dynamic quantization, and why not the alternatives:
  - rfdetr's ``export()`` has a ``quantization`` parameter, but its
    docstring says it is TFLite-only and "(ignored when format='onnx')".
    Passing quantization="int8" for an ONNX export silently produces a
    plain fp32 file -- there is no INT8 QDQ path in the rfdetr ONNX
    exporter (verified against rfdetr 1.9.1 source, detr.py).
  - The fp16 converter (onnxconverter-common) produces a graph ONNX
    Runtime rejects -- it inserts Cast nodes whose output types
    disagree with the graph's value_info, and neither boundary-Cast
    surgery nor shape-inference refresh fixes it.
  - The fp32 export is ~115 MB, over GitHub's 100 MB per-file push
    limit, so it cannot be committed to the repo. Dynamic
    quantization brings it well under the limit while keeping the
    model loadable by ONNX Runtime on the phone (same runtime the
    app already ships).

  2026-10-15 phone failure: the first dynamically quantized model
  FAILED to load on the phone with ORT_NOT_IMPLEMENTED ("Could not
  find an implementation for ConvInteger(10) node ...
  /patch_embeddings/projection/Conv_quant"). The app ships
  onnxruntime-android 1.22.0, whose build has no ConvInteger kernel,
  while the CI runner's newer Python ORT runs it fine. Root cause:
  onnxruntime 1.31's quantize_dynamic defaults quantize Conv as well
  (despite the docstring claim above that Conv stays fp32), producing
  ConvInteger ops. Fix: pass op_types_to_quantize=["MatMul", "Gemm",
  "Attention"] explicitly so no Conv is ever quantized, and assert
  after quantization that the graph contains no ConvInteger /
  QLinearConv nodes (phone-compat check). MatMul-only dynamic
  quantization uses DynamicQuantizeLinear + MatMul + DequantizeLinear,
  all supported by onnxruntime-android 1.22.0.
"""

import hashlib
import json
import sys
import time
from pathlib import Path


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    import verify_rfdetr

    root = Path(__file__).resolve().parent
    export_dir = root / "rfdetr_export"
    export_dir.mkdir(exist_ok=True)

    # 1. Export fp32 with Roboflow's own code.
    import importlib.metadata

    try:
        rfdetr_version = importlib.metadata.version("rfdetr")
    except Exception:
        rfdetr_version = "unknown"
    print(f"rfdetr version: {rfdetr_version}")

    from rfdetr import RFDETRNano

    model = RFDETRNano()
    model.export(str(export_dir), format="onnx")
    onnx_files = sorted(export_dir.glob("*.onnx"))
    print(f"exported files: {[(p.name, round(p.stat().st_size / 1e6, 1)) for p in onnx_files]}")
    if len(onnx_files) != 1:
        print(f"FAIL: expected exactly 1 fp32 ONNX file, got {[p.name for p in onnx_files]}")
        sys.exit(1)
    fp32_path = onnx_files[0]
    print(f"fp32: {fp32_path.name} ({fp32_path.stat().st_size / 1e6:.1f} MB)")

    # 2. Dynamic-quantize with ONNX Runtime's own tool.
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic

    print(f"onnxruntime version: {ort.__version__}")
    out_path = root / "rf-detr-nano.onnx"
    # MatMul-only: Conv must NOT be quantized. ORT 1.31's
    # quantize_dynamic defaults include Conv, which produces
    # ConvInteger ops that onnxruntime-android 1.22.0 (the phone
    # build) cannot run (ORT_NOT_IMPLEMENTED on-device 2026-10-15).
    quantize_dynamic(
        str(fp32_path),
        str(out_path),
        op_types_to_quantize=["MatMul", "Gemm", "Attention"],
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"quantized: {out_path.name} ({size_mb:.1f} MB)")
    if size_mb > 95:
        print(f"FAIL: quantized model {size_mb:.1f} MB is too close to GitHub's 100 MB file limit")
        sys.exit(1)

    # Phone-compat check: the quantized graph must contain no ops
    # that onnxruntime-android 1.22.0 cannot run. ConvInteger /
    # QLinearConv are the quantized-Conv ops; their presence means a
    # Conv got quantized and the model will fail to load on the phone.
    import onnx
    from collections import Counter
    graph = onnx.load(str(out_path)).graph
    op_counts = Counter(n.op_type for n in graph.node)
    print(f"quantized op types: {dict(op_counts)}")
    bad = sorted(set(op_counts) & {"ConvInteger", "QLinearConv"})
    if bad:
        print(f"FAIL: quantized model contains phone-incompatible ops: {bad}")
        sys.exit(1)
    print("phone-compat op check OK (no ConvInteger / QLinearConv)")

    # 3. Verify fp32 vs quantized on the bundled test image.
    image = root / "test-bus.jpg"
    fp32_res = verify_rfdetr.run_model(str(fp32_path), str(image), threshold=0.3)
    quant_res = verify_rfdetr.run_model(str(out_path), str(image), threshold=0.3)
    fp32_labels = [d["label"] for d in fp32_res["detections"]]
    quant_labels = [d["label"] for d in quant_res["detections"]]
    print(f"fp32 detections ({len(fp32_labels)}): {fp32_labels[:10]}")
    print(f"quant detections ({len(quant_labels)}): {quant_labels[:10]}")
    print(f"fp32 latency_ms mean: {fp32_res['latency_ms']['mean']}")
    print(f"quant latency_ms mean: {quant_res['latency_ms']['mean']}")
    if "bus" not in fp32_labels or "bus" not in quant_labels:
        print("FAIL: 'bus' not detected by fp32 and/or quantized on test-bus.jpg")
        sys.exit(1)
    # Parity: same SET of top-5 labels, scores within 0.1.
    # (Order may shuffle when two detections score close together.)
    fp32_top = {d["label"]: d["confidence"] for d in fp32_res["detections"][:5]}
    quant_top = {d["label"]: d["confidence"] for d in quant_res["detections"][:5]}
    if set(fp32_top) != set(quant_top):
        print(f"FAIL: fp32/quant top-5 label sets differ: {sorted(fp32_top)} vs {sorted(quant_top)}")
        sys.exit(1)
    for label in fp32_top:
        if abs(fp32_top[label] - quant_top[label]) > 0.1:
            print(f"FAIL: fp32/quant score drift on '{label}': "
                  f"{fp32_top[label]} vs {quant_top[label]}")
            sys.exit(1)
    print("fp32/quant parity OK")

    # 4. Provenance.
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    provenance = {
        "model": "RF-DETR Nano (RFDETRNano), COCO pretrained",
        "source": "Roboflow rfdetr package (github.com/roboflow/rf-detr), "
                  "official COCO checkpoint downloaded by RFDETRNano() from HuggingFace (roboflow org)",
        "rfdetr_version": rfdetr_version,
        "export": "model.export(format='onnx') with Roboflow's rfdetr ONNX exporter (opset 17, fp32), "
                  "then onnxruntime.quantization.quantize_dynamic with explicit "
                  "op_types_to_quantize=['MatMul', 'Gemm', 'Attention'] (MatMul/Gemm/Attention "
                  "weights to int8; Conv never quantized — phone ORT 1.22.0 has no ConvInteger "
                  "kernel). rfdetr's export() quantization parameter is "
                  "TFLite-only and ignored for ONNX; the fp32 export (~115 MB) exceeds GitHub's "
                  "100 MB file limit, so dynamic quantization is what makes the file committable.",
        "quantization": "dynamic int8, MatMul/Gemm/Attention only (onnxruntime.quantization.quantize_dynamic, "
                        f"onnxruntime {ort.__version__}); weights int8, activations fp32; "
                        "Conv stays fp32; phone-compat op check passed (no ConvInteger/QLinearConv)",
        "license": "Apache 2.0 (RF-DETR-N is Apache 2.0 per roboflow/rf-detr README benchmarks)",
        "file": "rf-detr-nano.onnx",
        "sha256": sha256_of(str(out_path)),
        "size_bytes": out_path.stat().st_size,
        "input": {"name": sess.get_inputs()[0].name,
                  "shape": [int(d) for d in sess.get_inputs()[0].shape]},
        "outputs": [{"name": o.name, "shape": [int(d) for d in o.shape]}
                    for o in sess.get_outputs()],
        "class_layout": "logits (Q, 91); slot index = COCO category ID (sparse 1-90); "
                        "official COCO weights: final slot is the real class 90, not background; "
                        "decode keeps every slot, unmapped slots never emit",
        "decode": "per-class sigmoid, top-Q pairs, score > 0.3, no NMS; "
                  "preprocess: direct resize, /255, ImageNet mean/std, NCHW",
        "verification": {
            "image": "test-bus.jpg",
            "fp32": {"detections": fp32_labels[:10],
                     "latency_ms_mean": fp32_res["latency_ms"]["mean"]},
            "quantized": {"detections": quant_labels[:10],
                          "latency_ms_mean": quant_res["latency_ms"]["mean"]},
            "parity": "same top-5 label set, scores within 0.1",
        },
        "exported_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    prov_path = root / "RFDETR_PROVENANCE.json"
    prov_path.write_text(json.dumps(provenance, indent=2))
    print(f"wrote {prov_path}")
    print("EXPORT OK")


if __name__ == "__main__":
    main()
