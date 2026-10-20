package com.battlesbudz.phoneinference

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.YuvImage
import android.os.Bundle
import android.util.Size
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.min

/**
 * Live Tracking mode: live camera preview + RF-DETR per frame, with a
 * ByteTrack-style tracker, a doorway line (IN/OUT counts), a floor zone
 * (in-zone count), trails, and a live FPS/counts bar. All on-device; no
 * cloud, no Roboflow API calls at runtime.
 *
 * Additive only: the localhost server (VisionService) is untouched.
 * The camera feed here never leaves the device.
 *
 * Line + zone are fixed to start: the line sits at 55% of frame height,
 * the zone is the trapezoid below it (the "floor"). Crossing the line
 * downward counts as IN, upward as OUT.
 */
class LiveTrackingActivity : ComponentActivity() {

    companion object {
        /** Detection confidence threshold, per the approved plan. */
        const val CONFIDENCE = 0.3f

        /** Line position as a fraction of frame height. */
        const val LINE_Y_NORM = 0.55f

        /** Floor zone polygon, normalized (0..1) coordinates. */
        val ZONE_NORM: List<Pt> = listOf(
            Pt(0.10f, 0.55f),
            Pt(0.90f, 0.55f),
            Pt(0.98f, 0.98f),
            Pt(0.02f, 0.98f),
        )
    }

    private lateinit var topBar: TextView
    private lateinit var statusView: TextView
    private lateinit var previewView: PreviewView
    private lateinit var trackingOverlay: TrackingOverlay
    private lateinit var toggleButton: Button

    private val tracker = LiveTracker()
    private val analysisExecutor = Executors.newSingleThreadExecutor()
    private val analyzing = AtomicBoolean(false)
    private var running = false
    private var fps = 0.0
    private var lastFrameNs = 0L

    private val cameraPermLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera()
            else statusView.text = "Camera permission denied; live tracking needs it."
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(buildLayout())
        toggleButton.setOnClickListener { toggle() }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            cameraPermLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onPause() {
        super.onPause()
        stopCamera()
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
    }

    private fun toggle() {
        if (running) stopCamera() else startCamera()
    }

    private fun startCamera() {
        if (running) return
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            try {
                val provider = providerFuture.get()
                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }
                val analysis = ImageAnalysis.Builder()
                    .setTargetResolution(Size(640, 480))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                    .build()
                    .also { it.setAnalyzer(analysisExecutor, ::onFrame) }
                provider.unbindAll()
                provider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    analysis,
                )
                running = true
                toggleButton.text = "Stop"
                statusView.text = "Live tracking running (RF-DETR Nano, on-device)."
            } catch (e: Exception) {
                statusView.text = "Could not start camera: ${e.message}"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun stopCamera() {
        if (!running) return
        try {
            ProcessCameraProvider.getInstance(this).get().unbindAll()
        } catch (_: Exception) {
        }
        running = false
        toggleButton.text = "Start"
        statusView.text = "Live tracking stopped."
    }

    private fun onFrame(image: ImageProxy) {
        if (!analyzing.compareAndSet(false, true)) {
            image.close()
            return
        }
        try {
            val bitmap = imageProxyToBitmap(image)
            // Detector choice mirrors VisionActivity: bundled RF-DETR first,
            // stock YOLOv8n fallback when it cannot load.
            val (detections, _) = try {
                val rfdetr = RfDetrDetectorHolder.get(this)
                if (rfdetr != null) {
                    rfdetr.detect(bitmap, CONFIDENCE, 300)
                } else {
                    val eng = EngineHolder.ensure(assets)
                    eng.detector.detect(bitmap, CONFIDENCE, 0.45f, 300)
                }
            } catch (e: Exception) {
                runOnUiThread { statusView.text = "Detection failed: ${e.message}" }
                emptyList<Detection>() to 0.0
            }
            val nowNs = System.nanoTime()
            if (lastFrameNs != 0L) {
                val dtMs = (nowNs - lastFrameNs) / 1e6
                if (dtMs > 0) {
                    val instant = 1000.0 / dtMs
                    fps = if (fps == 0.0) instant else 0.9 * fps + 0.1 * instant
                }
            }
            lastFrameNs = nowNs
            val result = tracker.update(
                detections,
                bitmap.width,
                bitmap.height,
                LINE_Y_NORM,
                ZONE_NORM,
            )
            val fpsSnap = fps
            runOnUiThread {
                trackingOverlay.setState(result.tracks, bitmap.width, bitmap.height)
                topBar.text =
                    "FPS %.1f · Now %d · IN %d · OUT %d · In zone %d".format(
                        fpsSnap,
                        result.tracks.size,
                        result.inCount,
                        result.outCount,
                        result.inZoneCount,
                    )
            }
            bitmap.recycle()
        } catch (e: Exception) {
            runOnUiThread { statusView.text = "Frame failed: ${e.message}" }
        } finally {
            analyzing.set(false)
            image.close()
        }
    }

    /** YUV_420_888 ImageProxy -> upright Bitmap. */
    private fun imageProxyToBitmap(proxy: ImageProxy): Bitmap {
        val image = proxy.image ?: throw IllegalStateException("Frame has no image")
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        val uvPixelStride = image.planes[1].pixelStride
        if (uvPixelStride == 1) {
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)
        } else {
            val uRowStride = image.planes[1].rowStride
            val vRowStride = image.planes[2].rowStride
            val w = image.width / 2
            val h = image.height / 2
            var out = ySize
            for (row in 0 until h) {
                for (col in 0 until w) {
                    nv21[out++] = vBuffer.get(row * vRowStride + col * uvPixelStride)
                    nv21[out++] = uBuffer.get(row * uRowStride + col * uvPixelStride)
                }
            }
        }
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val stream = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 90, stream)
        val bytes = stream.toByteArray()
        var bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
            ?: throw IllegalStateException("Could not decode camera frame")
        val rotation = proxy.imageInfo.rotationDegrees
        if (rotation != 0) {
            val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
            val rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)
            if (rotated != bmp) bmp.recycle()
            bmp = rotated
        }
        return bmp
    }

    // ---------------------------------------------------------- UI construction

    private fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()

    private fun buildLayout(): View {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        topBar = TextView(this).apply {
            text = "FPS — · Now — · IN 0 · OUT 0 · In zone 0"
            textSize = 13f
            typeface = android.graphics.Typeface.MONOSPACE
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(dp(8), dp(8), dp(8), dp(8))
        }
        statusView = TextView(this).apply {
            text = "Starting camera…"
            textSize = 12f
            gravity = Gravity.CENTER_HORIZONTAL
        }
        previewView = PreviewView(this).apply {
            scaleType = PreviewView.ScaleType.FIT_CENTER
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT,
            )
        }
        trackingOverlay = TrackingOverlay(this).apply {
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT,
            )
        }
        val frame = FrameLayout(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f
            )
            addView(previewView)
            addView(trackingOverlay)
        }
        toggleButton = Button(this).apply { text = "Stop" }
        val buttonRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_HORIZONTAL
            addView(toggleButton)
        }
        root.addView(topBar)
        root.addView(statusView)
        root.addView(frame)
        root.addView(buttonRow)
        return root
    }

    /**
     * Draws boxes, labels, IDs, trails, the line and the zone over the
     * preview. The preview is FIT_CENTER, so the overlay uses the same
     * letterbox math to map frame pixels onto view pixels.
     */
    private class TrackingOverlay(context: android.content.Context) : View(context) {
        private var tracks: List<TrackedObject> = emptyList()
        private var frameW = 1
        private var frameH = 1

        private val boxPaint = Paint().apply {
            style = Paint.Style.STROKE
            strokeWidth = 5f
            isAntiAlias = true
        }
        private val fillPaint = Paint().apply { style = Paint.Style.FILL }
        private val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = 42f
            isAntiAlias = true
        }
        private val linePaint = Paint().apply {
            color = Color.WHITE
            strokeWidth = 5f
            isAntiAlias = true
        }
        private val zoneFillPaint = Paint().apply {
            color = Color.argb(40, 255, 255, 255)
            style = Paint.Style.FILL
        }
        private val zoneBorderPaint = Paint().apply {
            color = Color.WHITE
            style = Paint.Style.STROKE
            strokeWidth = 4f
            isAntiAlias = true
        }
        private val trailPaint = Paint().apply {
            style = Paint.Style.STROKE
            strokeWidth = 5f
            isAntiAlias = true
        }

        fun setState(newTracks: List<TrackedObject>, w: Int, h: Int) {
            tracks = newTracks
            frameW = w
            frameH = h
            invalidate()
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            if (frameW <= 0 || frameH <= 0) return
            val scale = min(width.toFloat() / frameW, height.toFloat() / frameH)
            val ox = (width - frameW * scale) / 2f
            val oy = (height - frameH * scale) / 2f
            fun mx(x: Float) = ox + x * scale
            fun my(y: Float) = oy + y * scale

            // Zone (normalized polygon, mapped to view coords).
            val zonePath = Path()
            ZONE_NORM.forEachIndexed { i, p ->
                val x = mx(p.x * frameW)
                val y = my(p.y * frameH)
                if (i == 0) zonePath.moveTo(x, y) else zonePath.lineTo(x, y)
            }
            zonePath.close()
            canvas.drawPath(zonePath, zoneFillPaint)
            canvas.drawPath(zonePath, zoneBorderPaint)

            // Line at 55% of frame height.
            val lineY = my(LINE_Y_NORM * frameH)
            canvas.drawLine(mx(0f), lineY, mx(frameW.toFloat()), lineY, linePaint)

            for (track in tracks) {
                val inZone = track.inZone
                val color = if (inZone) Color.rgb(255, 152, 0) else Color.rgb(0, 200, 83)
                // Trail: fading polyline through the last centers.
                val trail = track.trail
                for (i in 1 until trail.size) {
                    val alpha = (255 * i / trail.size).coerceIn(0, 255)
                    trailPaint.color = Color.argb(alpha, Color.red(color), Color.green(color), Color.blue(color))
                    canvas.drawLine(
                        mx(trail[i - 1].x), my(trail[i - 1].y),
                        mx(trail[i].x), my(trail[i].y),
                        trailPaint,
                    )
                }
                // Box.
                val rect = RectF(
                    mx(track.box.x1), my(track.box.y1),
                    mx(track.box.x2), my(track.box.y2),
                )
                boxPaint.color = color
                canvas.drawRect(rect, boxPaint)
                // Label: "person #3 0.87".
                val label = "${track.label} #${track.id} ${"%.2f".format(track.confidence)}"
                val textWidth = textPaint.measureText(label)
                fillPaint.color = color
                canvas.drawRect(
                    rect.left, rect.top - 48f, rect.left + textWidth + 16f, rect.top, fillPaint
                )
                canvas.drawText(label, rect.left + 8f, rect.top - 8f, textPaint)
            }
        }
    }
}
