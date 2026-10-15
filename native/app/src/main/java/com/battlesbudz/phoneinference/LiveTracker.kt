package com.battlesbudz.phoneinference

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * ByteTrack-style multi-object tracker over per-frame [Detection]s, plus
 * LineZone and PolygonZone counting. Pure Kotlin, no Android dependencies,
 * so the counting logic is testable without a device.
 *
 * Matching: greedy by score, where score is IoU when boxes overlap and a
 * center-distance score otherwise. A track survives [MAX_MISSED] unmatched
 * frames before it is dropped, so a one-frame detector miss does not
 * renumber an object.
 *
 * Line crossing: the line is horizontal at [lineYNorm] * frame height.
 * A track whose center moves from above the line to below it counts as IN;
 * below to above counts as OUT. Each track counts once per crossing.
 *
 * Zone: a polygon in normalized (0..1) coordinates. A track counts as
 * in-zone when its box center is inside the polygon (ray casting).
 */
data class TrackedObject(
    val id: Int,
    val label: String,
    val confidence: Float,
    val box: Box,
    val trail: List<Pt>,
    val inZone: Boolean,
)

data class TrackerResult(
    val tracks: List<TrackedObject>,
    val inCount: Int,
    val outCount: Int,
    val inZoneCount: Int,
)

class LiveTracker {

    companion object {
        /** Frames a track may go unmatched before it is dropped. */
        const val MAX_MISSED = 10

        /** Max trail points kept per track. */
        const val MAX_TRAIL = 12

        /** Minimum IoU for a direct match. */
        const val MIN_IOU = 0.2f

        /** Max center distance (fraction of frame diagonal) for a fallback match. */
        const val MAX_CENTER_DIST_FRAC = 0.12f
    }

    private var nextId = 1
    private val tracks = mutableListOf<MutableTrack>()
    private var inCount = 0
    private var outCount = 0

    fun reset() {
        tracks.clear()
        nextId = 1
        inCount = 0
        outCount = 0
    }

    /**
     * Feed one frame of detections. [frameW]/[frameH] are the pixel size the
     * detection boxes are expressed in. [lineYNorm] is the horizontal line
     * position as a fraction of frame height. [zoneNorm] is the zone polygon
     * in normalized (0..1) coordinates. Returns the live tracks + counts.
     */
    fun update(
        detections: List<Detection>,
        frameW: Int,
        frameH: Int,
        lineYNorm: Float,
        zoneNorm: List<Pt>,
    ): TrackerResult {
        val lineY = lineYNorm * frameH
        val zone = zoneNorm.map { Pt(it.x * frameW, it.y * frameH) }
        val diag = kotlin.math.hypot(frameW.toFloat(), frameH.toFloat())

        // Greedy matching: score every (track, detection) pair, take the
        // best pairs first, each track and detection used at most once.
        data class Candidate(val trackIdx: Int, val detIdx: Int, val score: Float)
        val candidates = mutableListOf<Candidate>()
        for (ti in tracks.indices) {
            for (di in detections.indices) {
                val score = matchScore(tracks[ti].box, detections[di].box, diag)
                if (score > 0f) candidates.add(Candidate(ti, di, score))
            }
        }
        candidates.sortByDescending { it.score }
        val matchedTracks = mutableSetOf<Int>()
        val matchedDets = mutableSetOf<Int>()
        val assignments = mutableMapOf<Int, Int>() // trackIdx -> detIdx
        for (c in candidates) {
            if (c.trackIdx in matchedTracks || c.detIdx in matchedDets) continue
            matchedTracks.add(c.trackIdx)
            matchedDets.add(c.detIdx)
            assignments[c.trackIdx] = c.detIdx
        }

        // Update matched tracks; count line crossings from the center move.
        for ((ti, di) in assignments) {
            val track = tracks[ti]
            val det = detections[di]
            val newCenter = center(det.box)
            val newSide = if (newCenter.y < lineY) -1 else 1
            // Only count a crossing when the track was previously on the
            // other side. New tracks start with lastSide set below, so the
            // first sighting never counts.
            if (track.lastSide == -1 && newSide == 1) inCount++
            if (track.lastSide == 1 && newSide == -1) outCount++
            track.lastSide = newSide
            track.label = det.label
            track.confidence = det.confidence
            track.box = det.box
            track.missedFrames = 0
            track.trail.add(newCenter)
            while (track.trail.size > MAX_TRAIL) track.trail.removeAt(0)
        }

        // Unmatched tracks age out; drop ones missed too long.
        for (ti in tracks.indices) {
            if (ti !in assignments) tracks[ti].missedFrames++
        }
        tracks.removeAll { it.missedFrames > MAX_MISSED }

        // Unmatched detections start new tracks.
        for (di in detections.indices) {
            if (di in matchedDets) continue
            val det = detections[di]
            val c = center(det.box)
            val side = if (c.y < lineY) -1 else 1
            tracks.add(
                MutableTrack(
                    id = nextId++,
                    label = det.label,
                    confidence = det.confidence,
                    box = det.box,
                    trail = ArrayDeque(listOf(c)),
                    missedFrames = 0,
                    lastSide = side,
                )
            )
        }

        val out = tracks.map { t ->
            TrackedObject(
                id = t.id,
                label = t.label,
                confidence = t.confidence,
                box = t.box,
                trail = t.trail.toList(),
                inZone = pointInPolygon(center(t.box), zone),
            )
        }
        return TrackerResult(
            tracks = out,
            inCount = inCount,
            outCount = outCount,
            inZoneCount = out.count { it.inZone },
        )
    }

    /** IoU score, or a center-distance score when boxes do not overlap. */
    private fun matchScore(a: Box, b: Box, diag: Float): Float {
        val iou = iou(a, b)
        if (iou >= MIN_IOU) return iou
        val ca = center(a)
        val cb = center(b)
        val dist = kotlin.math.hypot(ca.x - cb.x, ca.y - cb.y)
        val maxDist = MAX_CENTER_DIST_FRAC * diag
        if (dist > maxDist) return 0f
        return max(0f, 1f - dist / maxDist) * 0.5f
    }

    private fun iou(a: Box, b: Box): Float {
        val x1 = max(a.x1, b.x1)
        val y1 = max(a.y1, b.y1)
        val x2 = min(a.x2, b.x2)
        val y2 = min(a.y2, b.y2)
        val w = x2 - x1
        val h = y2 - y1
        if (w <= 0f || h <= 0f) return 0f
        val inter = w * h
        val areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
        val areaB = (b.x2 - b.x1) * (b.y2 - b.y1)
        val union = areaA + areaB - inter
        if (union <= 0f) return 0f
        return inter / union
    }

    private fun center(box: Box): Pt =
        Pt((box.x1 + box.x2) / 2f, (box.y1 + box.y2) / 2f)

    /** Ray-casting point-in-polygon. Empty polygon = never inside. */
    internal fun pointInPolygon(p: Pt, poly: List<Pt>): Boolean {
        if (poly.size < 3) return false
        var inside = false
        var j = poly.size - 1
        for (i in poly.indices) {
            val xi = poly[i].x
            val yi = poly[i].y
            val xj = poly[j].x
            val yj = poly[j].y
            if ((yi > p.y) != (yj > p.y)) {
                val xIntersect = (p.y - yi) * (xj - xi) / (yj - yi) + xi
                if (p.x < xIntersect) inside = !inside
            }
            j = i
        }
        return inside
    }

    private class MutableTrack(
        val id: Int,
        var label: String,
        var confidence: Float,
        var box: Box,
        val trail: ArrayDeque<Pt>,
        var missedFrames: Int,
        var lastSide: Int,
    )
}
