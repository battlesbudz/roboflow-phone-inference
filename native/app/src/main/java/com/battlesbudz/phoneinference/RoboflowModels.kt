package com.battlesbudz.phoneinference

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKeys
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import org.json.JSONObject

/**
 * Manages an optional user-supplied Roboflow project model.
 *
 * The bundled stock weights (YOLOv8n etc.) stay the fallback so the app works
 * on first launch with zero setup. The DEFAULT detector is the bundled
 * RF-DETR Nano (Roboflow's own architecture, `models/rf-detr-nano.onnx`),
 * loaded by [RfDetrDetectorHolder]. From the Model settings screen the user
 * can switch between RF-DETR, stock, and a Roboflow project model
 * (downloaded via the Roboflow REST API, stored in app-private storage,
 * and marked active). The active detector is chosen by the detection mode
 * ([detectionMode]); [CustomDetectorHolder] handles the custom case.
 *
 * API surface used (verified against the roboflow-python SDK source and
 * https://docs.roboflow.com/models/model-weights/download-roboflow-model-weights):
 *   GET https://api.roboflow.com/{workspace}/{project}/{version}?api_key=...
 *       -> version info (validates key / names; may carry class labels)
 *   GET https://api.roboflow.com/{workspace}/{project}/{version}/onnx?api_key=...&nocache=true
 *       -> 200 {"export": {"link": "<signed download url>"}}
 *          202 {"progress": 0.0-1.0} while Roboflow prepares the export
 *
 * The API key lives in EncryptedSharedPreferences (Android Keystore backed)
 * and is never logged. Everything else lives in plain app-private prefs.
 */
object RoboflowModels {

    private const val API_BASE = "https://api.roboflow.com"
    private const val SECRET_PREFS = "roboflow_secrets"
    private const val MODEL_PREFS = "roboflow_model"
    private const val KEY_API_KEY = "api_key"
    private const val KEY_WORKSPACE = "workspace"
    private const val KEY_PROJECT = "project"
    private const val KEY_VERSION = "version"
    private const val KEY_DISPLAY_NAME = "display_name"
    private const val KEY_LABELS = "labels_json"
    private const val KEY_DETECTION_MODE = "detection_mode"
    private const val MODEL_FILE = "roboflow_model.onnx"

    /** Detection modes. RF-DETR (bundled) is the default; stock is the fallback. */
    const val MODE_RFDETR = "rfdetr"
    const val MODE_STOCK = "stock"
    const val MODE_CUSTOM = "custom"

    /** Plain-language failure; safe to show in the UI. Never carries the key. */
    class RoboflowException(message: String) : Exception(message)

    private fun secretPrefs(context: Context): SharedPreferences {
        // security-crypto 1.0.0 API: MasterKeys.getOrCreate returns the
        // Keystore alias; EncryptedSharedPreferences.create takes it first.
        val masterKeyAlias = MasterKeys.getOrCreate(MasterKeys.AES256_GCM_SPEC)
        return EncryptedSharedPreferences.create(
            SECRET_PREFS,
            masterKeyAlias,
            context,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
        )
    }

    private fun modelPrefs(context: Context): SharedPreferences =
        context.getSharedPreferences(MODEL_PREFS, Context.MODE_PRIVATE)

    fun saveApiKey(context: Context, apiKey: String) {
        secretPrefs(context).edit().putString(KEY_API_KEY, apiKey).apply()
    }

    fun getApiKey(context: Context): String? =
        secretPrefs(context).getString(KEY_API_KEY, null)?.takeIf { it.isNotBlank() }

    fun saveProject(context: Context, workspace: String, project: String, version: String) {
        modelPrefs(context).edit()
            .putString(KEY_WORKSPACE, workspace)
            .putString(KEY_PROJECT, project)
            .putString(KEY_VERSION, version)
            .apply()
    }

    fun projectTriple(context: Context): Triple<String, String, String> {
        val p = modelPrefs(context)
        return Triple(
            p.getString(KEY_WORKSPACE, "").orEmpty(),
            p.getString(KEY_PROJECT, "").orEmpty(),
            p.getString(KEY_VERSION, "").orEmpty(),
        )
    }

    /** Which detector is active. Defaults to RF-DETR (bundled). */
    fun detectionMode(context: Context): String =
        modelPrefs(context).getString(KEY_DETECTION_MODE, null) ?: MODE_RFDETR

    fun setDetectionMode(context: Context, mode: String) {
        require(mode in setOf(MODE_RFDETR, MODE_STOCK, MODE_CUSTOM)) {
            "Unknown detection mode: $mode"
        }
        modelPrefs(context).edit().putString(KEY_DETECTION_MODE, mode).apply()
    }

    /** Human-readable active model, reflecting the detection mode. */
    fun activeModelName(context: Context): String =
        when (detectionMode(context)) {
            MODE_RFDETR -> "RF-DETR Nano (Roboflow, bundled)"
            MODE_CUSTOM -> modelPrefs(context).getString(KEY_DISPLAY_NAME, null)
                ?: "Stock models (YOLOv8n + MobileNetV2)"
            else -> "Stock models (YOLOv8n + MobileNetV2)"
        }

    fun hasCustomModel(context: Context): Boolean =
        modelPrefs(context).getString(KEY_DISPLAY_NAME, null) != null &&
            modelFile(context).exists()

    fun modelFile(context: Context): File =
        File(context.filesDir, "roboflow/$MODEL_FILE")

    fun customLabels(context: Context): List<String> {
        val raw = modelPrefs(context).getString(KEY_LABELS, null) ?: return emptyList()
        return try {
            val arr = org.json.JSONArray(raw)
            List(arr.length()) { arr.getString(it) }
        } catch (_: Exception) {
            emptyList()
        }
    }

    fun clearCustomModel(context: Context) {
        modelPrefs(context).edit().clear().apply()
        modelFile(context).delete()
        // Keep the API key: the user may just be switching back to stock.
        CustomDetectorHolder.invalidate()
        // clear() wiped the mode too; stock is the fallback.
        setDetectionMode(context, MODE_STOCK)
    }

    private fun enc(segment: String): String =
        URLEncoder.encode(segment, "UTF-8").replace("+", "%20")

    private fun get(url: String): Pair<Int, String> {
        val conn = (URL(url).openConnection() as HttpURLConnection).apply {
            connectTimeout = 20_000
            readTimeout = 30_000
            setRequestProperty("User-Agent", "roboflow-phone-inference/1.0")
            instanceFollowRedirects = true
        }
        return try {
            val code = conn.responseCode
            val stream = if (code in 200..299) conn.inputStream else conn.errorStream
            val body = stream?.bufferedReader()?.readText().orEmpty()
            code to body
        } finally {
            conn.disconnect()
        }
    }

    /**
     * Downloads the configured project's ONNX weights and activates them.
     * Must run on a background thread. [onProgress] gets 0..1 updates.
     * @throws RoboflowException with a plain-language message on any failure.
     */
    @Throws(RoboflowException::class)
    fun downloadAndActivate(context: Context, onProgress: (Float) -> Unit) {
        val apiKey = getApiKey(context)
            ?: throw RoboflowException("No API key saved. Enter it on the Model screen first.")
        val (workspace, project, version) = projectTriple(context)
        if (workspace.isBlank() || project.isBlank() || version.isBlank()) {
            throw RoboflowException("Workspace, project and version are all required.")
        }
        val keyParam = "api_key=" + URLEncoder.encode(apiKey, "UTF-8")

        // 1. Version info: validates key + names, and may carry class labels.
        onProgress(0.02f)
        val infoUrl = "$API_BASE/${enc(workspace)}/${enc(project)}/${enc(version)}?$keyParam"
        val (infoCode, infoBody) = get(infoUrl)
        val labels: List<String> = when (infoCode) {
            200 -> extractLabels(infoBody)
            401 -> throw RoboflowException(
                "Roboflow rejected the API key. Double-check it in Settings \u2192 Model and try again."
            )
            404 -> throw RoboflowException(
                "Couldn't find $workspace/$project version $version on Roboflow. " +
                    "Check the workspace, project and version names."
            )
            else -> throw RoboflowException(
                "Roboflow returned an unexpected response ($infoCode) while checking the project version."
            )
        }

        // 2. Ask for the ONNX export; poll while Roboflow prepares it (HTTP 202).
        onProgress(0.08f)
        val exportUrl = "$API_BASE/${enc(workspace)}/${enc(project)}/${enc(version)}/onnx?$keyParam&nocache=true"
        var link: String? = null
        val deadline = System.currentTimeMillis() + 4 * 60_000L
        while (System.currentTimeMillis() < deadline) {
            val (code, body) = get(exportUrl)
            when (code) {
                200 -> {
                    link = try {
                        JSONObject(body).getJSONObject("export").getString("link")
                    } catch (_: Exception) {
                        throw RoboflowException(
                            "Roboflow's response didn't include a download link. The project may not offer an ONNX export."
                        )
                    }
                    break
                }
                202 -> {
                    val progress = try {
                        JSONObject(body).optDouble("progress", 0.0)
                    } catch (_: Exception) {
                        0.0
                    }
                    onProgress(0.08f + (0.10f * progress.coerceIn(0.0, 1.0).toFloat()))
                    Thread.sleep(3_000)
                }
                401 -> throw RoboflowException(
                    "Roboflow rejected the API key. Double-check it in Settings \u2192 Model and try again."
                )
                404 -> throw RoboflowException(
                    "Roboflow has no ONNX export for $workspace/$project version $version. " +
                        "Only some model types (e.g. YOLOv8/v11/v12, RF-DETR) can be downloaded as ONNX."
                )
                else -> throw RoboflowException(
                    "Roboflow returned an unexpected response ($code) while preparing the ONNX export."
                )
            }
        }
        val downloadUrl = link
            ?: throw RoboflowException(
                "Roboflow is still preparing the ONNX export. Wait a minute and try again."
            )

        // 3. Download the weights to app-private storage.
        onProgress(0.2f)
        val dir = modelFile(context).parentFile!!
        dir.mkdirs()
        val tmp = File(dir, "$MODEL_FILE.tmp")
        try {
            downloadFile(downloadUrl, tmp) { frac ->
                onProgress(0.2f + 0.75f * frac)
            }
            validateOnnx(tmp)
            if (!tmp.renameTo(modelFile(context))) {
                throw RoboflowException("Couldn't save the downloaded model file.")
            }
        } finally {
            tmp.delete()
        }

        // 4. Activate.
        onProgress(0.98f)
        modelPrefs(context).edit()
            .putString(KEY_DISPLAY_NAME, "$workspace/$project v$version")
            .putString(KEY_LABELS, org.json.JSONArray(labels).toString())
            .apply()
        setDetectionMode(context, MODE_CUSTOM)
        CustomDetectorHolder.invalidate()
        onProgress(1f)
    }

    private fun downloadFile(url: String, dest: File, onProgress: (Float) -> Unit) {
        val conn = (URL(url).openConnection() as HttpURLConnection).apply {
            connectTimeout = 20_000
            readTimeout = 60_000
            instanceFollowRedirects = true
        }
        try {
            val code = conn.responseCode
            if (code !in 200..299) {
                throw RoboflowException("Model download failed (HTTP $code). Check your connection and try again.")
            }
            val total = conn.contentLengthLong.takeIf { it > 0 }
            var done = 0L
            conn.inputStream.use { input ->
                FileOutputStream(dest).use { output ->
                    val buf = ByteArray(64 * 1024)
                    while (true) {
                        val n = input.read(buf)
                        if (n < 0) break
                        output.write(buf, 0, n)
                        done += n
                        if (total != null) onProgress((done.toFloat() / total).coerceIn(0f, 1f))
                    }
                }
            }
        } catch (e: RoboflowException) {
            throw e
        } catch (_: Exception) {
            throw RoboflowException("Model download failed. Check your connection and try again.")
        } finally {
            conn.disconnect()
        }
    }

    /** Rejects files that are clearly not ONNX weights (e.g. an HTML error page or a zip). */
    private fun validateOnnx(file: File) {
        val header = ByteArray(4)
        val n = file.inputStream().use { it.read(header) }
        if (n < 4 || (header[0] == 0x50.toByte() && header[1] == 0x4B.toByte())) {
            throw RoboflowException(
                "The downloaded file doesn't look like ONNX weights. " +
                    "Try the download again, or pick a different project version."
            )
        }
        if (file.length() < 1_000_000) {
            throw RoboflowException(
                "The downloaded file is suspiciously small (${file.length()} bytes). " +
                    "It probably isn't model weights; try again."
            )
        }
    }

    /**
     * Best-effort class-label extraction from version metadata. Roboflow
     * version payloads vary; accept a "classes" object or array and ignore
     * anything else. Empty list = fall back to generic "class N" labels.
     */
    private fun extractLabels(body: String): List<String> {
        return try {
            val root = JSONObject(body)
            if (!root.has("classes")) return emptyList()
            when (val c = root.get("classes")) {
                is JSONObject -> {
                    val keys = c.keys().asSequence().toList()
                    val intKeys = keys.mapNotNull { it.toIntOrNull() }
                    if (intKeys.size == keys.size) {
                        intKeys.sorted().map { c.getString(it.toString()) }
                    } else {
                        keys.sorted().map { c.getString(it) }
                    }
                }
                is org.json.JSONArray -> List(c.length()) { c.getString(it) }
                else -> emptyList()
            }
        } catch (_: Exception) {
            emptyList()
        }
    }
}
