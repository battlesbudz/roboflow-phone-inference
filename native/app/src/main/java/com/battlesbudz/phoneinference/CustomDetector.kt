package com.battlesbudz.phoneinference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import java.nio.FloatBuffer

/**
 * A user-supplied Roboflow project model running in the detection slot.
 *
 * Supports YOLO-style single-output detectors (YOLOv8/v11/v12 ONNX exports):
 * output0 is (1, 4 + numClasses, N) with cx,cy,w,h rows followed by one row
 * per class. Input size is read from the model's own input shape. Anything
 * else (e.g. RF-DETR, segmentation, classification exports) fails fast with
 * a plain-language message instead of producing garbage boxes.
 *
 * Output is List<Detection> in original-image pixels, exactly like the stock
 * [Detector], so the VisionActivity UI, the localhost /detect endpoint and
 * [CaptionBuilder] all work unchanged.
 */
class CustomDetector(
    private val env: OrtEnvironment,
    modelBytes: ByteArray,
    val labels: List<String>,
    val displayName: String,
) {
    private val session: OrtSession
    private val inputName: String
    val inputSize: Int

    /** Derived from the first output tensor; -1 until the first run. */
    private var numClasses: Int = -1

    init {
        val opts = OrtSession.SessionOptions()
        session = env.createSession(modelBytes, opts)
        inputName = session.inputNames.iterator().next()
        val shape = (session.inputInfo[inputName] as? ai.onnxruntime.TensorInfo)?.shape
            ?: throw IllegalArgumentException(
                "The model \"$displayName\" has an unexpected input; only image detectors are supported."
            )
        require(shape.size == 4 && shape[1] == 3L && shape[2] > 0 && shape[2] == shape[3]) {
            "The model \"$displayName\" doesn't take a square RGB image input; only YOLO-style detectors are supported."
        }
        inputSize = shape[2].toInt()
    }

    fun detect(
        bitmap: Bitmap,
        confidence: Float = 0.25f,
        iouThreshold: Float = 0.45f,
        maxDetections: Int = 300,
    ): Pair<List<Detection>, Double> {
        val lb = letterbox(bitmap, inputSize)
        val result = mutableListOf<Detection>()
        var inferMs = 0.0
        OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(lb.chw),
            longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong()),
        ).use { input ->
            val t0 = System.nanoTime()
            session.run(mapOf(inputName to input)).use { out ->
                inferMs = (System.nanoTime() - t0) / 1e6
                @Suppress("UNCHECKED_CAST")
                val m = (out[0].value as Array<Array<FloatArray>>)[0] // [4+C][N]
                val dims = m.size
                if (dims < 5) {
                    throw IllegalArgumentException(
                        "The model \"$displayName\" doesn't look like a YOLO-style detector " +
                            "(expected 4 + classes output rows, got $dims). " +
                            "Only YOLOv8/v11/v12 detection exports are supported."
                    )
                }
                val n = m[0].size
                val c = dims - 4
                if (numClasses < 0) numClasses = c
                val boxesCxcywh = Array(n) { FloatArray(4) }
                val scores = FloatArray(n)
                val classIds = IntArray(n)
                for (i in 0 until n) {
                    var best = 0
                    var bestScore = m[4][i]
                    for (cls in 1 until c) {
                        val s = m[4 + cls][i]
                        if (s > bestScore) { bestScore = s; best = cls }
                    }
                    scores[i] = bestScore
                    classIds[i] = best
                    for (k in 0..3) boxesCxcywh[i][k] = m[k][i]
                }
                val kept = (0 until n).filter { scores[it] >= confidence }
                val kb = Array(kept.size) { boxesCxcywh[kept[it]] }
                val ks = FloatArray(kept.size) { scores[kept[it]] }
                val xyxy = mapToOriginal(cxcywhToXyxy(kb), lb.scale, lb.padX, lb.padY, lb.origW, lb.origH)
                for (idx in nms(xyxy, ks, iouThreshold)) {
                    val gi = kept[idx]
                    val b = xyxy[idx]
                    val cls = classIds[gi]
                    result.add(
                        Detection(
                            label = labels.getOrElse(cls) { "class $cls" },
                            classId = cls,
                            confidence = round3(scores[gi]),
                            box = Box(round1(b[0]), round1(b[1]), round1(b[2]), round1(b[3])),
                        )
                    )
                }
            }
        }
        result.sortByDescending { it.confidence }
        return result.take(maxDetections) to inferMs
    }

    fun close() = session.close()
}
