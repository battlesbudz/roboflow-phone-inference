package com.battlesbudz.phoneinference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.res.AssetManager
import android.graphics.Bitmap
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * YOLOv8n-seg instance segmentation (yolov8n-seg.onnx).
 * Mirrors server.py `Segmenter.segment`:
 *   output0 (1, 116, 8400) = 4 box + 80 classes + 32 mask coefficients
 *   output1 (1, 32, 160, 160) mask prototypes
 * Mask = sigmoid(coeffs @ protos), cropped to the box, polygon via scanline trace.
 * Points are in original-image pixels. Returns (results, inference ms).
 */
class Segmenter(private val env: OrtEnvironment, assets: AssetManager) {
    private val session: OrtSession
    private val inputName: String

    init {
        val opts = OrtSession.SessionOptions()
        session = env.createSession(assets.open("models/yolov8n-seg.onnx").readBytes(), opts)
        inputName = session.inputNames.iterator().next()
    }

    fun segment(
        bitmap: Bitmap,
        confidence: Float = 0.4f,
        iouThreshold: Float = 0.3f,
        maxDetections: Int = 300,
    ): Pair<List<SegResult>, Double> {
        val lb = letterbox(bitmap, 640)
        val results = mutableListOf<SegResult>()
        var inferMs = 0.0
        OnnxTensor.createTensor(env, FloatBuffer.wrap(lb.chw), longArrayOf(1, 3, 640, 640)).use { input ->
            val t0 = System.nanoTime()
            session.run(mapOf(inputName to input)).use { out ->
                inferMs = (System.nanoTime() - t0) / 1e6
                @Suppress("UNCHECKED_CAST")
                val m = (out[0].value as Array<Array<FloatArray>>)[0] // [116][8400]
                @Suppress("UNCHECKED_CAST")
                val p = (out[1].value as Array<Array<Array<FloatArray>>>)[0] // [32][160][160]
                val n = 8400
                val numCls = 80
                val boxesCxcywh = Array(n) { FloatArray(4) }
                val scores = FloatArray(n)
                val labels = IntArray(n)
                val coeffs = Array(n) { FloatArray(32) }
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
                    for (c in 0 until 32) coeffs[i][c] = m[84 + c][i]
                }
                val kept = (0 until n).filter { scores[it] >= confidence }
                val kb = Array(kept.size) { boxesCxcywh[kept[it]] }
                val ks = FloatArray(kept.size) { scores[kept[it]] }
                val xyxyLb = cxcywhToXyxy(kb)
                val xyxy = mapToOriginal(xyxyLb, lb.scale, lb.padX, lb.padY, lb.origW, lb.origH)
                for (idx in nms(xyxy, ks, iouThreshold)) {
                    val gi = kept[idx]
                    val b = xyxy[idx]
                    val points = mutableListOf<Pt>()
                    // assemble mask at 160x160, crop to the letterboxed box
                    var bx1 = (xyxyLb[idx][0] / 4).toInt().coerceAtLeast(0)
                    var by1 = (xyxyLb[idx][1] / 4).toInt().coerceAtLeast(0)
                    var bx2 = (xyxyLb[idx][2] / 4).toInt().coerceAtMost(160)
                    var by2 = (xyxyLb[idx][3] / 4).toInt().coerceAtMost(160)
                    if (bx2 > bx1 && by2 > by1) {
                        val cw = bx2 - bx1
                        val ch = by2 - by1
                        val crop = Array(ch) { BooleanArray(cw) }
                        val coeff = coeffs[gi]
                        for (y in by1 until by2) {
                            for (x in bx1 until bx2) {
                                var s = 0f
                                for (c in 0 until 32) s += coeff[c] * p[c][y][x]
                                crop[y - by1][x - bx1] = 1f / (1f + exp(-s)) > 0.5f
                            }
                        }
                        for (pt in scanlinePolygon(crop)) {
                            val lx = (pt.x + bx1) * 4f
                            val ly = (pt.y + by1) * 4f
                            val ox = ((lx - lb.padX) / lb.scale).coerceIn(0f, lb.origW.toFloat())
                            val oy = ((ly - lb.padY) / lb.scale).coerceIn(0f, lb.origH.toFloat())
                            points.add(Pt(round1(ox), round1(oy)))
                        }
                    }
                    results.add(
                        SegResult(
                            label = COCO_LABELS[labels[gi]],
                            classId = labels[gi],
                            confidence = round3(scores[gi]),
                            box = Box(round1(b[0]), round1(b[1]), round1(b[2]), round1(b[3])),
                            points = points,
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

/**
 * Outline a binary mask as a polygon in mask-pixel coordinates.
 * Faithful port of server.py `_scanline_polygon`: trace down the left edge
 * of each non-empty row, back up the right edge. Holes are ignored.
 */
fun scanlinePolygon(binm: Array<BooleanArray>, maxPoints: Int = 200): List<Pt> {
    val left = mutableListOf<Pt>()
    val right = mutableListOf<Pt>()
    for (y in binm.indices) {
        val row = binm[y]
        var first = -1
        var last = -1
        for (x in row.indices) {
            if (row[x]) {
                if (first < 0) first = x
                last = x
            }
        }
        if (first < 0) continue
        left.add(Pt(first.toFloat(), y.toFloat()))
        right.add(Pt(last.toFloat(), y.toFloat()))
    }
    if (left.isEmpty()) return emptyList()
    var poly: List<Pt> = left + right.reversed()
    if (poly.size > maxPoints) {
        val step = poly.size.toDouble() / maxPoints
        poly = (0 until maxPoints).map { poly[(it * step).toInt()] }
    }
    return poly
}
