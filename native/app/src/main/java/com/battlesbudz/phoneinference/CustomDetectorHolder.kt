package com.battlesbudz.phoneinference

import ai.onnxruntime.OrtEnvironment
import android.content.Context
import android.util.Log

/**
 * Lazily loads the active Roboflow custom detector exactly once per process
 * (mirrors [EngineHolder] for the stock models). Returns null when the stock
 * models are active or the custom model can't be loaded, in which case
 * callers fall back to the stock detector.
 *
 * The API key is never logged here; failures log only the model name.
 */
object CustomDetectorHolder {
    private const val TAG = "CustomDetectorHolder"

    @Volatile
    private var detector: CustomDetector? = null

    @Volatile
    private var attemptedFor: String? = null

    @Volatile
    private var loadError: String? = null

    /**
     * Returns the active custom detector, loading it on first use.
     * Safe to call from a background thread; blocks while loading.
     */
    @Synchronized
    fun get(context: Context): CustomDetector? {
        val name = RoboflowModels.activeModelName(context)
        if (!RoboflowModels.hasCustomModel(context)) {
            return null
        }
        if (detector != null && attemptedFor == name) return detector
        if (attemptedFor == name) return null // already failed for this model
        attemptedFor = name
        loadError = null
        detector = try {
            val file = RoboflowModels.modelFile(context)
            val labels = RoboflowModels.customLabels(context)
            CustomDetector(OrtEnvironment.getEnvironment(), file.readBytes(), labels, name)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load custom model $name: ${e.message}")
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
        attemptedFor = null
        loadError = null
    }
}
