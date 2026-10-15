package com.battlesbudz.phoneinference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * RF-DETR Nano (Roboflow) object detection, bundled as `models/rf-detr-nano.onnx`.
 *
 * This is Roboflow's own architecture, exported with Roboflow's rfdetr code
 * from the official COCO checkpoint and dynamically quantized to int8 with
 * ONNX Runtime's quantize_dynamic. It is the default detector; the stock
 * YOLOv8n stays as the fallback when this model can't load.
 *
 * Decode mirrors Roboflow's own runtime
 * (rfdetr/export/_runtime/decode.py + _runtime/preprocess.py):
 *   - Preprocess: direct resize to the model's input size (NO letterbox),
 *     /255, ImageNet mean/std, NCHW float32.
 *   - Outputs: `dets` = (1, Q, 4) normalised cxcywh, `labels` = (1, Q, 91)
 *     raw logits. Matched by output name; falls back to positional 0/1.
 *   - Per-class sigmoid (logits clipped to [-88, 88]), keep EVERY class
 *     slot (the official COCO weights use the final slot as the real class
 *     90, not background), take the top-Q query/class pairs before
 *     thresholding, keep score > threshold. NO NMS: DETR queries do not
 *     duplicate boxes.
 *   - Boxes are cxcywh -> xyxy scaled by the ORIGINAL image size, clipped.
 *   - Slot index = COCO category ID (sparse 1-90) via
 *     [RFDETR_COCO_CATEGORY_TO_NAME]; unmapped slots never emit.
 *
 * Output is List<Detection> in original-image pixels, exactly like the stock
 * [Detector], so the VisionActivity UI, the localhost /detect endpoint and
 * [CaptionBuilder] all work unchanged.
 */
class RfDetrDetector(
    private val env: OrtEnvironment,
    modelBytes: ByteArray,
) {
    private val session: OrtSession
    private val inputName: String
    val inputSize: Int
    private val detsIndex: Int
    private val labelsIndex: Int

    init {
        val opts = OrtSession.SessionOptions()
        session = env.createSession(modelBytes, opts)
        inputName = session.inputNames.iterator().next()
        val shape = (session.inputInfo[inputName] as? ai.onnxruntime.TensorInfo)?.shape
            ?: throw IllegalArgumentException(
                "The RF-DETR model has an unexpected input; only image detectors are supported."
            )
        require(shape.size == 4 && shape[1] == 3L && shape[2] > 0 && shape[2] == shape[3]) {
            "The RF-DETR model doesn't take a square RGB image input."
        }
        inputSize = shape[2].toInt()
        val outNames = session.outputNames.toList()
        detsIndex = outNames.indexOfFirst { "dets" in it }.takeIf { it >= 0 } ?: 0
        labelsIndex = outNames.indexOfFirst { "labels" in it }.takeIf { it >= 0 } ?: 1
    }

    fun detect(
        bitmap: Bitmap,
        confidence: Float = 0.3f,
        maxDetections: Int = 300,
    ): Pair<List<Detection>, Double> {
        // Direct resize (no letterbox), /255, ImageNet normalize -> NCHW.
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val chw = FloatArray(3 * inputSize * inputSize)
        try {
            val pixels = IntArray(inputSize * inputSize)
            resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
            val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
            val std = floatArrayOf(0.229f, 0.224f, 0.225f)
            for (i in pixels.indices) {
                val p = pixels[i]
                val r = ((p shr 16) and 0xFF) / 255f
                val g = ((p shr 8) and 0xFF) / 255f
                val b = (p and 0xFF) / 255f
                chw[i] = (r - mean[0]) / std[0]
                chw[inputSize * inputSize + i] = (g - mean[1]) / std[1]
                chw[2 * inputSize * inputSize + i] = (b - mean[2]) / std[2]
            }
        } finally {
            resized.recycle()
        }

        val result = mutableListOf<Detection>()
        var inferMs = 0.0
        OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(chw),
            longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong()),
        ).use { input ->
            val t0 = System.nanoTime()
            session.run(mapOf(inputName to input)).use { out ->
                inferMs = (System.nanoTime() - t0) / 1e6
                @Suppress("UNCHECKED_CAST")
                val dets = (out[detsIndex].value as Array<Array<FloatArray>>)[0] // [Q][4]
                @Suppress("UNCHECKED_CAST")
                val labels = (out[labelsIndex].value as Array<Array<FloatArray>>)[0] // [Q][C]
                val q = dets.size
                val c = if (q > 0) labels[0].size else 0
                // Score every (query, slot) pair, then take the top-Q pairs:
                // score desc, then ascending flattened index (mirrors
                // Roboflow's topk, which is stable on the flattened order).
                val pairs = ArrayList<Pair<Float, Int>>(q * c)
                for (query in 0 until q) {
                    for (slot in 0 until c) {
                        val logit = labels[query][slot].coerceIn(-88f, 88f)
                        val score = 1f / (1f + exp(-logit))
                        pairs.add(score to query * c + slot)
                    }
                }
                pairs.sortWith(compareByDescending<Pair<Float, Int>> { it.first }.thenBy { it.second })
                val origW = bitmap.width.toFloat()
                val origH = bitmap.height.toFloat()
                var emitted = 0
                for ((score, flat) in pairs) {
                    if (emitted >= maxDetections) break
                    if (score <= confidence) break // pairs are score-descending
                    val query = flat / c
                    val slot = flat % c
                    val name = RFDETR_COCO_CATEGORY_TO_NAME[slot] ?: continue
                    val cx = dets[query][0]
                    val cy = dets[query][1]
                    val bw = dets[query][2]
                    val bh = dets[query][3]
                    val x1 = max(0f, (cx - bw / 2) * origW)
                    val y1 = max(0f, (cy - bh / 2) * origH)
                    val x2 = min(origW, (cx + bw / 2) * origW)
                    val y2 = min(origH, (cy + bh / 2) * origH)
                    result.add(
                        Detection(
                            label = name,
                            classId = slot,
                            confidence = round3(score),
                            box = Box(round1(x1), round1(y1), round1(x2), round1(y2)),
                        )
                    )
                    emitted++
                }
            }
        }
        // Already score-descending by construction.
        return result to inferMs
    }

    fun close() = session.close()
}
