package com.battlesbudz.phoneinference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.res.AssetManager
import android.graphics.Bitmap
import java.nio.FloatBuffer
import kotlin.math.exp

/**
 * MobileNetV2 ImageNet-1k classification (mobilenetv2-12.onnx).
 * Mirrors server.py `Classifier.classify`: direct 224x224 BILINEAR resize
 * (no letterbox), /255, ImageNet mean/std normalize, CHW, softmax top-5.
 */
class Classifier(private val env: OrtEnvironment, assets: AssetManager) {
    private val session: OrtSession
    private val inputName: String
    val labels: List<String>

    init {
        val opts = OrtSession.SessionOptions().apply {
            intraOpNumThreads = Runtime.getRuntime().availableProcessors()
        }
        session = env.createSession(assets.open("models/mobilenetv2-12.onnx").readBytes(), opts)
        inputName = session.inputNames.iterator().next()
        labels = assets.open("synset.txt").bufferedReader().readLines()
            .mapNotNull { line ->
                val t = line.trim()
                if (t.isEmpty()) null else t.substringAfter(" ")
            }
    }

    fun classify(bitmap: Bitmap): Pair<List<ClassPrediction>, Double> {
        val size = 224
        val resized = Bitmap.createScaledBitmap(bitmap, size, size, true)
        val pixels = IntArray(size * size)
        resized.getPixels(pixels, 0, size, 0, 0, size, size)
        if (resized !== bitmap) resized.recycle()

        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        val plane = size * size
        val chw = FloatArray(3 * plane)
        for (i in pixels.indices) {
            val p = pixels[i]
            val r = ((p shr 16) and 0xFF) / 255f
            val g = ((p shr 8) and 0xFF) / 255f
            val b = (p and 0xFF) / 255f
            chw[i] = (r - mean[0]) / std[0]
            chw[plane + i] = (g - mean[1]) / std[1]
            chw[2 * plane + i] = (b - mean[2]) / std[2]
        }

        var inferMs = 0.0
        var top5: List<ClassPrediction> = emptyList()
        OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), longArrayOf(1, 3, 224, 224)).use { input ->
            val t0 = System.nanoTime()
            session.run(mapOf(inputName to input)).use { out ->
                inferMs = (System.nanoTime() - t0) / 1e6
                @Suppress("UNCHECKED_CAST")
                val logits = (out[0].value as Array<FloatArray>)[0] // [1000]
                val maxLogit = logits.max()
                var sum = 0.0
                val exps = DoubleArray(logits.size) { k ->
                    exp((logits[k] - maxLogit).toDouble()).also { sum += it }
                }
                val top = logits.indices.sortedByDescending { exps[it] / sum }.take(5)
                top5 = top.map { k ->
                    ClassPrediction(
                        label = labels.getOrElse(k) { "class_$k" },
                        classId = k,
                        confidence = round4((exps[k] / sum).toFloat()),
                    )
                }
            }
        }
        return top5 to inferMs
    }

    fun close() = session.close()
}
