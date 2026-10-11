package com.battlesbudz.phoneinference

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Typeface
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import java.io.File
import kotlin.concurrent.thread
import kotlin.math.round

/**
 * Standalone vision app. Take or pick a photo and get a fused
 * natural-language description from the four on-device models.
 *
 * No CAMERA permission: capture goes through the system camera app and the
 * gallery through the Storage Access Framework; both return a usable Uri.
 * The old benchmark sweep lives behind the "Benchmark" toggle and the
 * localhost HTTP server behind "Server".
 */
class VisionActivity : ComponentActivity() {

    private lateinit var statusView: TextView
    private lateinit var preview: ImageView
    private lateinit var answerView: TextView
    private lateinit var photoButton: Button
    private lateinit var galleryButton: Button
    private lateinit var sampleButton: Button
    private lateinit var benchmarkPane: LinearLayout
    private lateinit var benchmarkOutput: TextView
    private lateinit var benchmarkButton: Button
    private lateinit var serverButton: Button
    private lateinit var serverInfo: TextView

    private var testBitmap: Bitmap? = null
    private var currentBitmap: Bitmap? = null
    private var pendingCaptureUri: Uri? = null
    private var benchmarkVisible = false
    private var serverOn = false

    private val captureLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == RESULT_OK) {
                pendingCaptureUri?.let { onImageUri(it) }
            }
        }
    private val pickLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == RESULT_OK) {
                result.data?.data?.let { onImageUri(it) }
            }
        }
    private val notifPermLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startVisionService()
            else setStatus("Notification permission denied; the server needs it for its persistent notification.")
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(buildLayout())

        photoButton.setOnClickListener { takePhoto() }
        galleryButton.setOnClickListener { pickFromGallery() }
        sampleButton.setOnClickListener { useSamplePhoto() }
        benchmarkButton.setOnClickListener { toggleBenchmark() }
        serverButton.setOnClickListener { toggleServer() }

        setStatus("Loading models…")
        thread {
            try {
                val eng = EngineHolder.ensure(assets)
                val bmp = assets.open("test-bus.jpg").use { BitmapFactory.decodeStream(it) }
                testBitmap = bmp
                val sb = StringBuilder("Models loaded:")
                for ((name, ms) in eng.loadMs) {
                    sb.append("\n  $name: ${r1(ms)} ms")
                }
                runOnUiThread {
                    setStatus(sb.toString())
                    photoButton.isEnabled = true
                    galleryButton.isEnabled = true
                    sampleButton.isEnabled = true
                }
            } catch (e: Exception) {
                runOnUiThread { setStatus("Failed to load models: ${e.message}") }
            }
        }
    }

    // ---------------------------------------------------------- UI construction

    private fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()

    private fun buildLayout(): View {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(20), dp(20), dp(20))
        }
        val title = TextView(this).apply {
            text = "Phone Inference"
            textSize = 20f
            setTypeface(typeface, Typeface.BOLD)
            gravity = Gravity.CENTER_HORIZONTAL
        }
        statusView = TextView(this).apply { textSize = 12f }
        preview = ImageView(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(220)
            )
            scaleType = ImageView.ScaleType.FIT_CENTER
            setBackgroundColor(0xFFEEEEEE.toInt())
        }
        val buttonRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_HORIZONTAL
        }
        photoButton = Button(this).apply { text = "Take photo"; isEnabled = false }
        galleryButton = Button(this).apply { text = "Gallery"; isEnabled = false }
        sampleButton = Button(this).apply { text = "Sample"; isEnabled = false }
        buttonRow.addView(photoButton)
        buttonRow.addView(galleryButton)
        buttonRow.addView(sampleButton)

        val answerLabel = TextView(this).apply {
            text = "What I see"
            setTypeface(typeface, Typeface.BOLD)
            textSize = 14f
        }
        answerView = TextView(this).apply {
            text = "Take a photo, choose one from the gallery, or try the sample."
            textSize = 14f
        }
        val answerScroll = ScrollView(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f
            )
            addView(answerView)
        }

        val toggleRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_HORIZONTAL
        }
        benchmarkButton = Button(this).apply { text = "Benchmark" }
        serverButton = Button(this).apply { text = "Start server" }
        toggleRow.addView(benchmarkButton)
        toggleRow.addView(serverButton)

        serverInfo = TextView(this).apply {
            textSize = 12f
            typeface = Typeface.MONOSPACE
            visibility = View.GONE
        }

        benchmarkPane = buildBenchmarkPane().apply { visibility = View.GONE }

        root.addView(title)
        root.addView(statusView)
        root.addView(preview)
        root.addView(buttonRow)
        root.addView(answerLabel)
        root.addView(answerScroll)
        root.addView(toggleRow)
        root.addView(serverInfo)
        root.addView(benchmarkPane)
        return root
    }

    private fun buildBenchmarkPane(): LinearLayout {
        val pane = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        val label = TextView(this).apply {
            text = "Benchmark (same sweep as the old screen)"
            setTypeface(typeface, Typeface.BOLD)
        }
        val runButton = Button(this).apply { text = "Run full sweep" }
        benchmarkOutput = TextView(this).apply {
            typeface = Typeface.MONOSPACE
            textSize = 12f
        }
        val scroll = ScrollView(this).apply { addView(benchmarkOutput) }
        runButton.setOnClickListener { runSweep(runButton) }
        pane.addView(label)
        pane.addView(runButton)
        pane.addView(scroll)
        return pane
    }

    // ---------------------------------------------------------- image sources

    private fun takePhoto() {
        val photoFile = File(cacheDir, "capture_${System.currentTimeMillis()}.jpg")
        val uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", photoFile)
        pendingCaptureUri = uri
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            .putExtra(MediaStore.EXTRA_OUTPUT, uri)
            .addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
        captureLauncher.launch(intent)
    }

    private fun pickFromGallery() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            type = "image/*"
            addCategory(Intent.CATEGORY_OPENABLE)
        }
        pickLauncher.launch(intent)
    }

    private fun useSamplePhoto() {
        thread {
            val bmp = try {
                assets.open("test-bus.jpg").use { BitmapFactory.decodeStream(it) }
            } catch (e: Exception) {
                runOnUiThread { setStatus("Sample photo missing: ${e.message}") }
                return@thread
            }
            runOnUiThread { setImage(bmp) }
        }
    }

    /** Decode a content Uri, downscaling so the long edge is at most 1280 px. */
    private fun decodeUri(uri: Uri): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null
        var sample = 1
        while (bounds.outWidth / sample > 1280 || bounds.outHeight / sample > 1280) {
            sample *= 2
        }
        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
        return contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) }
    }

    private fun onImageUri(uri: Uri) {
        setStatus("Loading image…")
        thread {
            val bmp = decodeUri(uri)
            if (bmp == null) {
                runOnUiThread { setStatus("Could not read that image.") }
                return@thread
            }
            runOnUiThread { setImage(bmp) }
        }
    }

    private fun setImage(bmp: Bitmap) {
        currentBitmap?.recycle()
        currentBitmap = bmp
        preview.setImageBitmap(bmp)
        describe(bmp)
    }

    // ---------------------------------------------------------- inference

    private fun describe(bitmap: Bitmap) {
        setStatus("Analyzing…")
        answerView.text = "…"
        thread {
            try {
                val eng = EngineHolder.ensure(assets)
                eng.detector.detect(bitmap) // warmup: first run includes kernel init
                val (dets, detMs) = eng.detector.detect(bitmap, 0.25f, 0.45f, 300)
                val (segs, segMs) = eng.segmenter.segment(bitmap, 0.25f, 0.45f, 300)
                val (poses, poseMs) = eng.poseDetector.pose(bitmap, 0.25f, 0.45f, 0f, 300)
                val (top5, clsMs) = eng.classifier.classify(bitmap)
                val caption = CaptionBuilder.build(dets, segs, poses, top5)
                val timings = "detection ${dets.size} (${r1(detMs)} ms) · " +
                    "segmentation ${segs.size} (${r1(segMs)} ms) · " +
                    "pose ${poses.size} (${r1(poseMs)} ms) · " +
                    "classification (${r1(clsMs)} ms)"
                runOnUiThread {
                    answerView.text = caption
                    setStatus(timings)
                }
            } catch (e: Exception) {
                runOnUiThread { setStatus("Analysis failed: ${e.message}") }
            }
        }
    }

    // ---------------------------------------------------------- benchmark pane

    private fun toggleBenchmark() {
        benchmarkVisible = !benchmarkVisible
        benchmarkPane.visibility = if (benchmarkVisible) View.VISIBLE else View.GONE
        benchmarkButton.text = if (benchmarkVisible) "Hide benchmark" else "Benchmark"
    }

    private fun runSweep(runButton: Button) {
        val eng = EngineHolder.get() ?: return
        val bmp = testBitmap ?: return
        runButton.isEnabled = false
        benchmarkOutput.text = "Running sweep…"
        thread {
            val sb = StringBuilder()
            try {
                eng.detector.detect(bmp) // warmup
                val (dets, detMs) = eng.detector.detect(bmp, 0.4f, 0.3f, 5)
                sb.append("\n== object_detection ==\n")
                sb.append("  ${dets.size} predictions, ${r1(detMs)} ms (baseline ~285 ms)\n")
                for (d in dets.take(3)) {
                    sb.append("    ${d.label.padEnd(12)} ${"%.3f".format(d.confidence)}\n")
                }
                val (segs, segMs) = eng.segmenter.segment(bmp, 0.4f, 0.3f, 3)
                sb.append("\n== instance_segmentation ==\n")
                sb.append("  ${segs.size} predictions, ${r1(segMs)} ms (baseline ~239 ms)\n")
                if (segs.isNotEmpty()) {
                    sb.append("    first mask: ${segs[0].points.size} polygon points, label=${segs[0].label}\n")
                }
                val (poses, poseMs) = eng.poseDetector.pose(bmp, 0.4f, 0.3f, 0f, 3)
                sb.append("\n== keypoint_detection ==\n")
                sb.append("  ${poses.size} predictions, ${r1(poseMs)} ms (baseline ~137 ms)\n")
                if (poses.isNotEmpty()) {
                    sb.append("    first person: ${poses[0].keypoints.size} keypoints\n")
                }
                val (top5, clsMs) = eng.classifier.classify(bmp)
                sb.append("\n== classification ==\n")
                sb.append("  top=${top5.firstOrNull()?.label} " +
                    "conf=${"%.4f".format(top5.firstOrNull()?.confidence ?: 0f)}, " +
                    "${r1(clsMs)} ms (baseline ~50 ms)\n")
            } catch (e: Exception) {
                sb.append("\nERROR: ${e.message}\n${e.stackTraceToString().take(2000)}")
            }
            val report = sb.toString()
            runOnUiThread {
                benchmarkOutput.text = report
                runButton.isEnabled = true
            }
        }
    }

    // ---------------------------------------------------------- local server

    private fun toggleServer() {
        if (serverOn) {
            stopService(Intent(this, VisionService::class.java))
            serverOn = false
            serverButton.text = "Start server"
            serverInfo.visibility = View.GONE
        } else {
            if (Build.VERSION.SDK_INT >= 33 &&
                ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) !=
                PackageManager.PERMISSION_GRANTED
            ) {
                notifPermLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
                return
            }
            startVisionService()
        }
    }

    private fun startVisionService() {
        ContextCompat.startForegroundService(this, Intent(this, VisionService::class.java))
        serverOn = true
        serverButton.text = "Stop server"
        val token = VisionService.tokenOf(this)
        serverInfo.text =
            "Listening on http://127.0.0.1:${VisionService.PORT}\n" +
                "Header: X-Vision-Token: $token"
        serverInfo.visibility = View.VISIBLE
    }

    // ---------------------------------------------------------- misc

    private fun setStatus(s: String) {
        statusView.text = s
    }

    private fun r1(ms: Double): String = "%.1f".format(round(ms * 10) / 10.0)

    override fun onDestroy() {
        // Engine is process-shared with the service; do not close it here.
        testBitmap?.recycle()
        currentBitmap?.recycle()
        super.onDestroy()
    }
}
