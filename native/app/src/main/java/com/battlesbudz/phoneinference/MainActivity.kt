package com.battlesbudz.phoneinference

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Typeface
import android.os.Bundle
import android.view.Gravity
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import kotlin.concurrent.thread

/**
 * Benchmark screen. Loads the bundled test image (assets/test-bus.jpg, the same
 * file the repo's test_all.py uses) and runs the same four-endpoint sweep the
 * Termux server was measured with.
 *
 * Compare the reported inference_ms numbers against the Termux baselines
 * (Galaxy Z Fold 6, 2026-10-08):
 *   detection      ~285 ms
 *   segmentation   ~239 ms
 *   pose           ~137 ms
 *   classification ~ 50 ms
 *
 * inference_ms here is pure model time (session.run), same definition as the
 * server's inference_ms field — no HTTP, no base64, no bitmap decode.
 */
class MainActivity : Activity() {

    private lateinit var output: TextView
    private lateinit var runButton: Button
    private var engine: InferenceEngine? = null
    private var testBitmap: Bitmap? = null
    private var loadReport: String = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(36, 36, 36, 36)
        }
        val title = TextView(this).apply {
            text = "Phone Inference — native ONNX benchmark"
            textSize = 18f
            setTypeface(typeface, Typeface.BOLD)
            gravity = Gravity.CENTER_HORIZONTAL
        }
        runButton = Button(this).apply {
            text = "Run full sweep"
            isEnabled = false
        }
        output = TextView(this).apply {
            text = "Loading models…"
            typeface = Typeface.MONOSPACE
            textSize = 12f
        }
        val scroll = ScrollView(this).apply { addView(output) }
        layout.addView(title)
        layout.addView(runButton)
        layout.addView(scroll)
        setContentView(layout)

        runButton.setOnClickListener { runSweep() }

        thread {
            try {
                val eng = InferenceEngine(assets)
                val bmp = assets.open("test-bus.jpg").use { BitmapFactory.decodeStream(it) }
                val sb = StringBuilder("Models loaded:\n")
                for ((name, ms) in eng.loadMs) {
                    sb.append("  %-14s load %7.1f ms\n".format(name, ms))
                }
                sb.append("\nImage: test-bus.jpg ${bmp.width}x${bmp.height}\nReady.")
                engine = eng
                testBitmap = bmp
                loadReport = sb.toString()
                runOnUiThread {
                    output.text = loadReport
                    runButton.isEnabled = true
                }
            } catch (e: Exception) {
                runOnUiThread { output.text = "Failed to load: ${e.message}\n${e.stackTraceToString().take(2000)}" }
            }
        }
    }

    private fun runSweep() {
        val eng = engine ?: return
        val bmp = testBitmap ?: return
        runButton.isEnabled = false
        output.text = "$loadReport\n\nRunning sweep…"
        thread {
            val sb = StringBuilder()
            try {
                // Warmup: first session.run includes one-time kernel init.
                eng.detector.detect(bmp)

                val (dets, detMs) = eng.detector.detect(bmp, 0.4f, 0.3f, 5)
                sb.append("\n== object_detection ==\n")
                sb.append("  ${dets.size} predictions, ${"%.1f".format(detMs)} ms (baseline ~285 ms)\n")
                for (d in dets.take(3)) {
                    sb.append("    ${d.label.padEnd(12)} ${"%.3f".format(d.confidence)}\n")
                }

                val (segs, segMs) = eng.segmenter.segment(bmp, 0.4f, 0.3f, 3)
                sb.append("\n== instance_segmentation ==\n")
                sb.append("  ${segs.size} predictions, ${"%.1f".format(segMs)} ms (baseline ~239 ms)\n")
                if (segs.isNotEmpty()) {
                    sb.append("    first mask: ${segs[0].points.size} polygon points, label=${segs[0].label}\n")
                }

                val (poses, poseMs) = eng.poseDetector.pose(bmp, 0.4f, 0.3f, 0f, 3)
                sb.append("\n== keypoint_detection ==\n")
                sb.append("  ${poses.size} predictions, ${"%.1f".format(poseMs)} ms (baseline ~137 ms)\n")
                if (poses.isNotEmpty()) {
                    sb.append("    first person: ${poses[0].keypoints.size} keypoints\n")
                }

                val (top5, clsMs) = eng.classifier.classify(bmp)
                sb.append("\n== classification ==\n")
                sb.append("  top=${top5.firstOrNull()?.label} " +
                        "conf=${"%.4f".format(top5.firstOrNull()?.confidence ?: 0f)}, " +
                        "${"%.1f".format(clsMs)} ms (baseline ~50 ms)\n")
            } catch (e: Exception) {
                sb.append("\nERROR: ${e.message}\n${e.stackTraceToString().take(2000)}")
            }
            val report = loadReport + sb.toString()
            runOnUiThread {
                output.text = report
                runButton.isEnabled = true
            }
        }
    }

    override fun onDestroy() {
        engine?.close()
        testBitmap?.recycle()
        super.onDestroy()
    }
}
