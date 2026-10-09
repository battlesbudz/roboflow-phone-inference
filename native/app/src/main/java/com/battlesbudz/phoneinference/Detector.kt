package com.battlesbudz.phoneinference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.res.AssetManager
import android.graphics.Bitmap
import java.nio.FloatBuffer
import kotlin.math.round

/**
 * YOLOv8n object detection (yolov8n.onnx).
 * Mirrors server.py `Detector.detect`: output0 is (1, 84, 8400) = cx,cy,w,h + 80 classes.
 * Returns (detections sorted by confidence desc, inference ms).
 */
class Detector(private val env: OrtEnvironment, assets: AssetManager) {
    private val session: OrtSession
    private val inputName: String

    init {
        val opts = OrtSession.SessionOptions()
        session = env.createSession(assets.open("models/yolov8n.onnx").readBytes(), opts)
        inputName = session.inputNames.iterator().next()
    }

    fun detect(
        bitmap: Bitmap,
        confidence: Float = 0.25f,
        iouThreshold: Float = 0.45f,
        maxDetections: Int = 300,
    ): Pair<List<Detection>, Double> {
        val lb = letterbox(bitmap, 640)
        val result = mutableListOf<Detection>()
        var inferMs = 0.0
        OnnxTensor.createTensor(env, FloatBuffer.wrap(lb.chw), longArrayOf(1, 3, 640, 640)).use { input ->
            val t0 = System.nanoTime()
            session.run(mapOf(inputName to input)).use { out ->
                inferMs = (System.nanoTime() - t0) / 1e6
                @Suppress("UNCHECKED_CAST")
                val m = (out[0].value as Array<Array<FloatArray>>)[0] // [84][8400]
                val n = 8400
                val numCls = 80
                val boxesCxcywh = Array(n) { FloatArray(4) }
                val scores = FloatArray(n)
                val labels = IntArray(n)
                for (i in 0 until n) {
                    var best = 0
                    var bestScore = m[4][i]
                    for (c in 1 until numCls) {
                        val s = m[4 + c][i]
                        if (s > bestScore) { bestScore = s; best = c }
                    }
                    scores[i] = bestScore
                    labels[i] = best
                    for (k in 0..3) boxesCxcywh[i][k] = m[k][i]
                }
                val kept = (0 until n).filter { scores[it] >= confidence }
                val kb = Array(kept.size) { boxesCxcywh[kept[it]] }
                val ks = FloatArray(kept.size) { scores[kept[it]] }
                val xyxy = mapToOriginal(cxcywhToXyxy(kb), lb.scale, lb.padX, lb.padY, lb.origW, lb.origH)
                for (idx in nms(xyxy, ks, iouThreshold)) {
                    val gi = kept[idx]
                    val b = xyxy[idx]
                    result.add(
                        Detection(
                            label = COCO_LABELS[labels[gi]],
                            classId = labels[gi],
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

internal fun round1(v: Float): Float = round(v * 10) / 10f
internal fun round3(v: Float): Float = round(v * 1000) / 1000f
internal fun round4(v: Float): Float = round(v * 10000) / 10000f
