package com.battlesbudz.phoneinference

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import kotlin.math.min

/**
 * Letterbox resize to a square input, mirroring server.py `_letterbox`:
 * BILINEAR resize, gray (114,114,114) canvas, centered paste, /255 NCHW float tensor.
 *
 * One known approximation: PIL's BILINEAR filter and Android's filtered
 * createScaledBitmap are both bilinear but not bit-identical, so tensor
 * values may differ in the last decimal places vs the Python server.
 */
data class LetterboxResult(
    val chw: FloatArray,
    val scale: Float,
    val padX: Int,
    val padY: Int,
    val origW: Int,
    val origH: Int,
)

fun letterbox(src: Bitmap, inputSize: Int = 640): LetterboxResult {
    val origW = src.width
    val origH = src.height
    val scale = min(inputSize / origW.toFloat(), inputSize / origH.toFloat())
    val newW = (origW * scale).toInt()
    val newH = (origH * scale).toInt()
    val resized = Bitmap.createScaledBitmap(src, newW, newH, true)
    val canvas = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
    canvas.eraseColor(Color.rgb(114, 114, 114))
    val padX = (inputSize - newW) / 2
    val padY = (inputSize - newH) / 2
    Canvas(canvas).drawBitmap(resized, padX.toFloat(), padY.toFloat(), null)
    if (resized !== src) resized.recycle()

    val pixels = IntArray(inputSize * inputSize)
    canvas.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
    canvas.recycle()

    val plane = inputSize * inputSize
    val chw = FloatArray(3 * plane)
    for (i in pixels.indices) {
        val p = pixels[i]
        chw[i] = ((p shr 16) and 0xFF) / 255f
        chw[plane + i] = ((p shr 8) and 0xFF) / 255f
        chw[2 * plane + i] = (p and 0xFF) / 255f
    }
    return LetterboxResult(chw, scale, padX, padY, origW, origH)
}

/** Map letterboxed xyxy boxes back to original-image pixels, clipped. Mirrors `_map_to_original`. */
fun mapToOriginal(
    xyxy: Array<FloatArray>,
    scale: Float,
    padX: Int,
    padY: Int,
    origW: Int,
    origH: Int,
): Array<FloatArray> {
    return Array(xyxy.size) { i ->
        floatArrayOf(
            ((xyxy[i][0] - padX) / scale).coerceIn(0f, origW.toFloat()),
            ((xyxy[i][1] - padY) / scale).coerceIn(0f, origH.toFloat()),
            ((xyxy[i][2] - padX) / scale).coerceIn(0f, origW.toFloat()),
            ((xyxy[i][3] - padY) / scale).coerceIn(0f, origH.toFloat()),
        )
    }
}

/** cxcywh -> xyxy in the same coordinate space. Mirrors `_xyxy_from_cxcywh`. */
fun cxcywhToXyxy(boxes: Array<FloatArray>): Array<FloatArray> {
    return Array(boxes.size) { i ->
        val (cx, cy, w, h) = boxes[i]
        floatArrayOf(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    }
}
