package com.battlesbudz.phoneinference

import android.app.Activity
import android.graphics.Typeface
import android.os.Bundle
import android.text.InputType
import android.view.Gravity
import android.view.WindowManager
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import kotlin.concurrent.thread

/**
 * "Bring your Roboflow project to your phone."
 *
 * The default detector is the bundled RF-DETR Nano (Roboflow's own
 * architecture, `models/rf-detr-nano.onnx`). From here the user can switch
 * to the stock models or enter a Roboflow API key + workspace + project +
 * version; the app then downloads that project's ONNX weights
 * ([RoboflowModels]) and runs them in the detection slot, fully offline
 * afterwards. The active model is always shown.
 *
 * The API key is stored in EncryptedSharedPreferences (Android Keystore
 * backed) and is never logged. FLAG_SECURE keeps the key out of screenshots
 * and the recent-apps thumbnail.
 */
class ModelSettingsActivity : Activity() {

    private lateinit var statusView: TextView
    private lateinit var keyField: EditText
    private lateinit var workspaceField: EditText
    private lateinit var projectField: EditText
    private lateinit var versionField: EditText
    private lateinit var progressView: TextView
    private lateinit var downloadButton: Button
    private lateinit var stockButton: Button
    private lateinit var rfdetrButton: Button

    private var busy = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.setFlags(
            WindowManager.LayoutParams.FLAG_SECURE,
            WindowManager.LayoutParams.FLAG_SECURE,
        )
        setContentView(buildLayout())
        refreshStatus()

        downloadButton.setOnClickListener { downloadClicked() }
        stockButton.setOnClickListener { stockClicked() }
        rfdetrButton.setOnClickListener { rfdetrClicked() }
    }

    override fun onResume() {
        super.onResume()
        refreshStatus()
    }

    private fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()

    private fun buildLayout(): ScrollView {
        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(20), dp(20), dp(20))
        }
        val title = TextView(this).apply {
            text = "Roboflow model"
            textSize = 20f
            setTypeface(typeface, Typeface.BOLD)
            gravity = Gravity.CENTER_HORIZONTAL
        }
        statusView = TextView(this).apply { textSize = 13f }
        val explainer = TextView(this).apply {
            text = "The bundled RF-DETR Nano (Roboflow's own model) is the default " +
                "detector. Switch to the stock models, or use one of your Roboflow " +
                "projects as the detector: the app downloads its ONNX weights once, " +
                "then runs fully offline. Your API key is stored encrypted on this " +
                "phone and never leaves it except to talk to Roboflow."
            textSize = 12f
        }

        keyField = EditText(this).apply {
            hint = if (RoboflowModels.getApiKey(this@ModelSettingsActivity) != null)
                "API key (saved)"
            else
                "API key (from roboflow.com/settings/api)"
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
        }
        workspaceField = EditText(this).apply { hint = "Workspace" }
        projectField = EditText(this).apply { hint = "Project" }
        versionField = EditText(this).apply {
            hint = "Version number (e.g. 3)"
            inputType = InputType.TYPE_CLASS_NUMBER
        }
        val (ws, proj, ver) = RoboflowModels.projectTriple(this)
        workspaceField.setText(ws)
        projectField.setText(proj)
        versionField.setText(ver)

        downloadButton = Button(this).apply { text = "Download & use this model" }
        rfdetrButton = Button(this).apply { text = "Use RF-DETR Nano (Roboflow, bundled)" }
        stockButton = Button(this).apply { text = "Use stock models" }
        progressView = TextView(this).apply { textSize = 12f }

        col.addView(title)
        col.addView(statusView)
        col.addView(explainer)
        col.addView(keyField)
        col.addView(workspaceField)
        col.addView(projectField)
        col.addView(versionField)
        col.addView(downloadButton)
        col.addView(rfdetrButton)
        col.addView(stockButton)
        col.addView(progressView)
        return ScrollView(this).apply { addView(col) }
    }

    private fun refreshStatus() {
        val active = RoboflowModels.activeModelName(this)
        val err = when (RoboflowModels.detectionMode(this)) {
            RoboflowModels.MODE_RFDETR -> RfDetrDetectorHolder.lastError()
            RoboflowModels.MODE_CUSTOM -> CustomDetectorHolder.lastError()
            else -> null
        }
        statusView.text = buildString {
            append("Active model: $active")
            if (err != null) append("\nNote: $active failed to load ($err); stock is being used.")
        }
    }

    private fun setBusy(b: Boolean) {
        busy = b
        downloadButton.isEnabled = !b
        rfdetrButton.isEnabled = !b
        stockButton.isEnabled = !b
    }

    private fun downloadClicked() {
        if (busy) return
        val keyInput = keyField.text.toString().trim()
        val ws = workspaceField.text.toString().trim()
        val proj = projectField.text.toString().trim()
        val ver = versionField.text.toString().trim()
        if (keyInput.isEmpty() && RoboflowModels.getApiKey(this) == null) {
            progressView.text = "Enter your Roboflow API key first."
            return
        }
        if (ws.isEmpty() || proj.isEmpty() || ver.isEmpty()) {
            progressView.text = "Workspace, project and version are all required."
            return
        }
        if (keyInput.isNotEmpty()) RoboflowModels.saveApiKey(this, keyInput)
        RoboflowModels.saveProject(this, ws, proj, ver)
        // Don't keep the key in the field any longer than needed.
        keyField.text.clear()

        setBusy(true)
        progressView.text = "Contacting Roboflow…"
        thread {
            try {
                RoboflowModels.downloadAndActivate(this) { frac ->
                    runOnUiThread {
                        progressView.text = "Downloading model… ${(frac * 100).toInt()}%"
                    }
                }
                runOnUiThread {
                    setBusy(false)
                    progressView.text = "Done."
                    Toast.makeText(this, "Now using $ws/$proj v$ver", Toast.LENGTH_LONG).show()
                    setResult(RESULT_OK)
                    finish()
                }
            } catch (e: RoboflowModels.RoboflowException) {
                runOnUiThread {
                    setBusy(false)
                    progressView.text = e.message
                }
            } catch (e: Exception) {
                runOnUiThread {
                    setBusy(false)
                    progressView.text = "Something went wrong: ${e.message}"
                }
            }
        }
    }

    private fun rfdetrClicked() {
        if (busy) return
        RoboflowModels.setDetectionMode(this, RoboflowModels.MODE_RFDETR)
        refreshStatus()
        progressView.text = "Switched to the bundled RF-DETR Nano."
        Toast.makeText(this, "Using RF-DETR Nano (Roboflow, bundled)", Toast.LENGTH_SHORT).show()
    }

    private fun stockClicked() {
        if (busy) return
        RoboflowModels.clearCustomModel(this)
        refreshStatus()
        progressView.text = "Switched back to the stock models."
        Toast.makeText(this, "Using stock models", Toast.LENGTH_SHORT).show()
    }
}
