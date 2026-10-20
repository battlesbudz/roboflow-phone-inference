package com.battlesbudz.phoneinference

import ai.onnxruntime.OrtEnvironment
import android.content.Context
import android.util.Log

/**
 * Lazily loads the bundled RF-DETR Nano detector exactly once per process
 * (mirrors [CustomDetectorHolder] for the user's project model). Returns
 * null when the bundled model can't be loaded, in which case callers fall
 * back to the stock YOLOv8n detector.
 */
object RfDetrDetectorHolder {
    private const val TAG = "RfDetrDetectorHolder"

    @Volatile
    private var detector: RfDetrDetector? = null

    @Volatile
    private var attempted: Boolean = false

    @Volatile
    private var loadError: String? = null

    /**
     * Returns the bundled RF-DETR detector, loading it on first use.
     * Safe to call from a background thread; blocks while loading.
     */
    @Synchronized
    fun get(context: Context): RfDetrDetector? {
        if (detector != null) return detector
        if (attempted) return null // already failed; don't retry every frame
        attempted = true
        loadError = null
        detector = try {
            val bytes = context.assets.open("models/rf-detr-nano.onnx").readBytes()
            RfDetrDetector(OrtEnvironment.getEnvironment(), bytes)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load bundled RF-DETR model: ${e.message}")
            loadError = e.message
            null
        }
        return detector
    }

    /** Why the last load failed, if it did. Safe to show in the UI. */
    fun lastError(): String? = loadError

    @Synchronized
    fun invalidate() {
        try {
            detector?.close()
        } catch (_: Exception) {
        }
        detector = null
        attempted = false
        loadError = null
    }
}
