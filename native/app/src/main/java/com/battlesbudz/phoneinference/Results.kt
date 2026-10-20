package com.battlesbudz.phoneinference

/** Result containers. Coordinates are in original-image pixels, matching server.py. */

data class Box(val x1: Float, val y1: Float, val x2: Float, val y2: Float)

data class Detection(
    val label: String,
    val classId: Int,
    val confidence: Float,
    val box: Box,
)

data class Pt(val x: Float, val y: Float)

data class SegResult(
    val label: String,
    val classId: Int,
    val confidence: Float,
    val box: Box,
    val points: List<Pt>,
)

data class Keypoint(
    val name: String,
    val classId: Int,
    val x: Float,
    val y: Float,
    val confidence: Float,
)

data class PoseResult(
    val confidence: Float,
    val box: Box,
    val keypoints: List<Keypoint>,
)

data class ClassPrediction(
    val label: String,
    val classId: Int,
    val confidence: Float,
)
