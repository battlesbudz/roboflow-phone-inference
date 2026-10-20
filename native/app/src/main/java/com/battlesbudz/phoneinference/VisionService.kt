package com.battlesbudz.phoneinference

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.BitmapFactory
import android.os.Build
import android.os.IBinder
import android.util.Base64
import android.util.Log
import fi.iki.elonen.NanoHTTPD
import org.json.JSONArray
import org.json.JSONObject
import java.security.SecureRandom
import kotlin.concurrent.thread
import kotlin.math.round

/**
 * Foreground service exposing the vision models over localhost HTTP so any
 * on-device agent can POST an image and get detections back.
 *
 *   GET  /health -> { status, model, version, model_load_ms, models }
 *   POST /detect -> { image: base64, confidence? } ->
 *                   { detections: [{label, confidence, box}], inference_ms }
 *
 * The server binds 127.0.0.1 only. Every request must carry the
 * X-Vision-Token header; the token is generated once, stored in
 * SharedPreferences, and shown in the app UI.
 */
class VisionService : Service() {

    companion object {
        const val PORT = 8080
        private const val CHANNEL_ID = "vision_server"
        private const val TAG = "VisionService"
        private const val PREFS = "vision_server_prefs"
        private const val KEY_TOKEN = "token"

        fun tokenOf(ctx: Context): String {
            val prefs = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            return prefs.getString(KEY_TOKEN, null) ?: generateToken().also {
                prefs.edit().putString(KEY_TOKEN, it).apply()
            }
        }

        private fun generateToken(): String {
            val bytes = ByteArray(24)
            SecureRandom().nextBytes(bytes)
            return Base64.encodeToString(bytes, Base64.URL_SAFE or Base64.NO_WRAP)
        }
    }

    private var server: NanoHTTPD? = null

    override fun onCreate() {
        super.onCreate()
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "Vision server", NotificationManager.IMPORTANCE_LOW)
        )
        val notification: Notification = Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("Phone Inference server running")
            .setContentText("Listening on 127.0.0.1:$PORT")
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .build()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(1, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        } else {
            @Suppress("DEPRECATION")
            startForeground(1, notification)
        }
        thread(name = "vision-server-startup") {
            try {
                EngineHolder.ensure(assets)
                server = VisionHttpd().also { it.start() }
                Log.i(TAG, "listening on 127.0.0.1:$PORT")
            } catch (e: Exception) {
                Log.e(TAG, "failed to start server", e)
                stopSelf()
            }
        }
    }

    override fun onDestroy() {
        try {
            server?.stop()
        } catch (_: Exception) {
        }
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private inner class VisionHttpd : NanoHTTPD("127.0.0.1", PORT) {
        private val token = tokenOf(this@VisionService)

        override fun serve(session: IHTTPSession): Response {
            return try {
                when {
                    session.method == Method.GET && session.uri == "/health" ->
                        jsonResponse(Response.Status.OK, healthJson())
                    session.method == Method.POST && session.uri == "/detect" ->
                        handleDetect(session)
                    else ->
                        jsonResponse(Response.Status.NOT_FOUND, """{"error":"not found"}""")
                }
            } catch (e: Exception) {
                Log.e(TAG, "serve error", e)
                jsonResponse(Response.Status.INTERNAL_ERROR, """{"error":"internal error"}""")
            }
        }

        private fun handleDetect(session: IHTTPSession): Response {
            // NanoHTTPD lower-cases header names.
            if (session.headers["x-vision-token"] != token) {
                return jsonResponse(
                    Response.Status.UNAUTHORIZED,
                    """{"error":"missing or invalid X-Vision-Token"}"""
                )
            }
            val files = HashMap<String, String>()
            session.parseBody(files)
            val body = files["postData"]
                ?: return jsonResponse(Response.Status.BAD_REQUEST, """{"error":"empty body"}""")
            val req = try {
                JSONObject(body)
            } catch (_: Exception) {
                return jsonResponse(Response.Status.BAD_REQUEST, """{"error":"body is not JSON"}""")
            }
            val b64 = req.optString("image", "")
            if (b64.isBlank()) {
                return jsonResponse(Response.Status.BAD_REQUEST, """{"error":"image (base64) is required"}""")
            }
            val bytes = try {
                Base64.decode(b64, Base64.DEFAULT)
            } catch (_: IllegalArgumentException) {
                return jsonResponse(Response.Status.BAD_REQUEST, """{"error":"image is not valid base64"}""")
            }
            val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                ?: return jsonResponse(Response.Status.BAD_REQUEST, """{"error":"image could not be decoded"}""")
            val engine = EngineHolder.get()
                ?: return jsonResponse(Response.Status.SERVICE_UNAVAILABLE, """{"error":"models still loading"}""")
            val confidence = req.optDouble("confidence", 0.25).toFloat()
            // Same active-detector selection as VisionActivity: a downloaded
            // Roboflow project model when one is active, else stock YOLOv8n.
            val custom = CustomDetectorHolder.get(this@VisionService)
            val (detections, inferMs) = if (custom != null) {
                custom.detect(bitmap, confidence, 0.45f, 300)
            } else {
                engine.detector.detect(bitmap, confidence, 0.45f, 300)
            }
            val arr = JSONArray()
            for (d in detections) {
                arr.put(
                    JSONObject()
                        .put("label", d.label)
                        .put("confidence", d.confidence.toDouble())
                        .put(
                            "box", JSONObject()
                                .put("x1", d.box.x1.toDouble())
                                .put("y1", d.box.y1.toDouble())
                                .put("x2", d.box.x2.toDouble())
                                .put("y2", d.box.y2.toDouble())
                        )
                )
            }
            val resp = JSONObject()
                .put("detections", arr)
                .put("inference_ms", round(inferMs * 10) / 10.0)
            bitmap.recycle()
            return jsonResponse(Response.Status.OK, resp.toString())
        }

        /** Mirrors the shape of server.py GET /health. */
        private fun healthJson(): String {
            val engine = EngineHolder.get()
            val loadMs = engine?.loadMs ?: emptyMap()
            val models = JSONObject()
            val entries = listOf(
                "object_detection" to ("yolov8n.onnx" to "detection"),
                "instance_segmentation" to ("yolov8n-seg.onnx" to "segmentation"),
                "keypoint_detection" to ("yolov8n-pose.onnx" to "pose"),
                "classification" to ("mobilenetv2-12.onnx" to "classification"),
            )
            for ((api, pair) in entries) {
                val (file, key) = pair
                models.put(
                    api, JSONObject()
                        .put("file", file)
                        .put("load_ms", loadMs[key] ?: -1.0)
                )
            }
            return JSONObject()
                .put("status", "ok")
                .put(
                    "model",
                    when (RoboflowModels.detectionMode(this@VisionService)) {
                        RoboflowModels.MODE_RFDETR -> "rf-detr"
                        RoboflowModels.MODE_CUSTOM ->
                            RoboflowModels.activeModelName(this@VisionService)
                        else -> "yolov8n"
                    },
                )
                .put("active_detector", RoboflowModels.activeModelName(this@VisionService))
                .put("version", BuildConfig.VERSION_NAME)
                .put("model_load_ms", loadMs["detection"] ?: -1.0)
                .put("models", models)
                .toString()
        }

        private fun jsonResponse(status: Response.Status, body: String): Response =
            newFixedLengthResponse(status, "application/json", body)
    }
}
