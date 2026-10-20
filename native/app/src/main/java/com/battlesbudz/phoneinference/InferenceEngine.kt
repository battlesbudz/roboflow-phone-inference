package com.battlesbudz.phoneinference

import ai.onnxruntime.OrtEnvironment
import android.content.res.AssetManager

/**
 * Owns the ORT environment and the four model sessions.
 * Same four models as the Termux server (server.py):
 *   detector      -> yolov8n.onnx        (/detect, /infer/object_detection)
 *   segmenter     -> yolov8n-seg.onnx    (/infer/instance_segmentation)
 *   poseDetector  -> yolov8n-pose.onnx   (/infer/keypoint_detection)
 *   classifier    -> mobilenetv2-12.onnx  (/infer/classification)
 */
class InferenceEngine(assets: AssetManager) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()

    val detector: Detector
    val segmenter: Segmenter
    val poseDetector: PoseDetector
    val classifier: Classifier

    /** Wall-clock ms to construct each session, mirroring server.py load_ms. */
    val loadMs: Map<String, Double>

    init {
        val times = mutableMapOf<String, Double>()
        var t0 = System.nanoTime()
        detector = Detector(env, assets)
        times["detection"] = (System.nanoTime() - t0) / 1e6
        t0 = System.nanoTime()
        segmenter = Segmenter(env, assets)
        times["segmentation"] = (System.nanoTime() - t0) / 1e6
        t0 = System.nanoTime()
        poseDetector = PoseDetector(env, assets)
        times["pose"] = (System.nanoTime() - t0) / 1e6
        t0 = System.nanoTime()
        classifier = Classifier(env, assets)
        times["classification"] = (System.nanoTime() - t0) / 1e6
        loadMs = times
    }

    fun close() {
        detector.close()
        segmenter.close()
        poseDetector.close()
        classifier.close()
        env.close()
    }
}
