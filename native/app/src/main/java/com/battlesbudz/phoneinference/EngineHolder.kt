package com.battlesbudz.phoneinference

import android.content.res.AssetManager

/**
 * Process-wide shared inference engine. The activity and the localhost HTTP
 * service both use this so the ~54 MB of models is loaded exactly once per
 * process. Loading is synchronous and takes a few seconds; call [ensure]
 * from a background thread.
 */
object EngineHolder {
    @Volatile
    private var engine: InferenceEngine? = null

    @Synchronized
    fun ensure(assets: AssetManager): InferenceEngine {
        return engine ?: InferenceEngine(assets).also { engine = it }
    }

    fun get(): InferenceEngine? = engine
}
