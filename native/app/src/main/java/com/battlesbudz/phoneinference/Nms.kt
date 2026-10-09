package com.battlesbudz.phoneinference

import kotlin.math.max
import kotlin.math.min

/**
 * Greedy class-agnostic NMS. Mirrors server.py `_nms` exactly
 * (boxes: xyxy, scores; keeps indices in descending-score order).
 */
fun nms(boxes: Array<FloatArray>, scores: FloatArray, iouThresh: Float = 0.45f): List<Int> {
    if (boxes.isEmpty()) return emptyList()
    val order = scores.indices.sortedByDescending { scores[it] }.toMutableList()
    val keep = mutableListOf<Int>()
    while (order.isNotEmpty()) {
        val i = order.removeAt(0)
        keep.add(i)
        if (order.isEmpty()) break
        val it = order.iterator()
        while (it.hasNext()) {
            val j = it.next()
            val xx1 = max(boxes[i][0], boxes[j][0])
            val yy1 = max(boxes[i][1], boxes[j][1])
            val xx2 = min(boxes[i][2], boxes[j][2])
            val yy2 = min(boxes[i][3], boxes[j][3])
            val w = max(0f, xx2 - xx1)
            val h = max(0f, yy2 - yy1)
            val inter = w * h
            val areaI = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
            val areaJ = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
            val iou = inter / (areaI + areaJ - inter + 1e-6f)
            if (iou > iouThresh) it.remove()
        }
    }
    return keep
}
