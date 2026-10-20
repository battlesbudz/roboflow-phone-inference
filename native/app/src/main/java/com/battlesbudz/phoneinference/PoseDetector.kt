package com.battlesbudz.phoneinference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.res.AssetManager
import android.graphics.Bitmap
import java.nio.FloatBuffer

/**
 * YOLOv8n-pose keypoint detection (yolov8n-pose.onnx).
 * Mirrors server.py `PoseDetector.pose`:
 *   output0 (1, 56, 8400) = 4 box + 1 person score + 17 keypoints * (x, y, conf).
 * Keypoints are in letterboxed 640 coords in the model output, mapped here
 * to original-image pixels. Returns (results, inference ms).
 */
class PoseDetector(private val env: OrtEnvironment, assets: AssetManager) {
    private val session: OrtSession
    private val inputName: String

    init {
        val opts = OrtSession.SessionOptions()
        session = env.createSession(assets.open("models/yolov8n-pose.onnx").readBytes(), opts)
        inputName = session.inputNames.iterator().next()
    }

    fun pose(
        bitmap: Bitmap,
        confidence: Float = 0.4f,
        iouThreshold: Float = 0.3f,
        keypointConfidence: Float = 0f,
        maxDetections: Int = 300,
    ): Pair<List<PoseResult>, Double> {
        val lb = letterbox(bitmap, 640)
        val results = mutableListOf<PoseResult>()
        var inferMs = 0.0
        OnnxTensor.createTensor(env, FloatBuffer.wrap(lb.chw), longArrayOf(1, 3, 640, 640)).use { input ->
            val t0 = System.nanoTime()
            session.run(mapOf(inputName to input)).use { out ->
                inferMs = (System.nanoTime() - t0) / 1e6
                @Suppress("UNCHECKED_CAST")
                val m = (out[0].value as Array<Array<FloatArray>>)[0] // [56][8400]
                val n = 8400
                val boxesCxcywh = Array(n) { FloatArray(4) }
                val scores = FloatArray(n)
                val kpts = Array(n) { Array(17) { FloatArray(3) } }
                for (i in 0 until n) {
                    scores[i] = m[4][i]
                    for (k in 0..3) boxesCxcywh[i][k] = m[k][i]
                    for (j in 0 until 17) {
                        kpts[i][j][0] = m[5 + j * 3][i]
                        kpts[i][j][1] = m[5 + j * 3 + 1][i]
                        kpts[i][j][2] = m[5 + j * 3 + 2][i]
                    }
                }
                val kept = (0 until n).filter { scores[it] >= confidence }
                val kb = Array(kept.size) { boxesCxcywh[kept[it]] }
                val ks = FloatArray(kept.size) { scores[kept[it]] }
                val xyxy = mapToOriginal(cxcywhToXyxy(kb), lb.scale, lb.padX, lb.padY, lb.origW, lb.origH)
                for (idx in nms(xyxy, ks, iouThreshold)) {
                    val gi = kept[idx]
                    val b = xyxy[idx]
                    val keypoints = mutableListOf<Keypoint>()
                    for (j in 0 until 17) {
                        val kc = kpts[gi][j][2]
                        if (kc < keypointConfidence) continue
                        val ox = ((kpts[gi][j][0] - lb.padX) / lb.scale).coerceIn(0f, lb.origW.toFloat())
                        val oy = ((kpts[gi][j][1] - lb.padY) / lb.scale).coerceIn(0f, lb.origH.toFloat())
                        keypoints.add(
                            Keypoint(
                                name = COCO_KEYPOINT_NAMES[j],
                                classId = j,
                                x = round1(ox),
                                y = round1(oy),
                                confidence = round3(kc),
                            )
                        )
                    }
                    results.add(
                        PoseResult(
                            confidence = round3(scores[gi]),
                            box = Box(round1(b[0]), round1(b[1]), round1(b[2]), round1(b[3])),
                            keypoints = keypoints,
                        )
                    )
                }
            }
        }
        results.sortByDescending { it.confidence }
        return results.take(maxDetections) to inferMs
    }

    fun close() = session.close()
}
