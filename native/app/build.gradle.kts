plugins {
    id("com.android.application")
    kotlin("android")
}

// The .onnx models, synset.txt and test-bus.jpg live at the repo root (shared
// with the Termux Python server). They are copied into a generated assets
// directory at build time so the APK stays self-contained without duplicating
// ~54 MB of binaries in git.
// NOTE: no trailing-lambda on from() — Kotlin DSL rejects `from(a, b) { }`
// ("Passing value as a vararg is only allowed inside a parenthesized
// argument list"). One Copy task per destination instead.
val copyModels = tasks.register<Copy>("copyModels") {
    from(
        rootProject.file("../yolov8n.onnx"),
        rootProject.file("../yolov8n-seg.onnx"),
        rootProject.file("../yolov8n-pose.onnx"),
        rootProject.file("../mobilenetv2-12.onnx")
    )
    into(layout.buildDirectory.dir("generated/assets/models"))
}
val copyModelExtras = tasks.register<Copy>("copyModelExtras") {
    from(
        rootProject.file("../synset.txt"),
        rootProject.file("../test-bus.jpg")
    )
    into(layout.buildDirectory.dir("generated/assets"))
}

android {
    namespace = "com.battlesbudz.phoneinference"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.battlesbudz.phoneinference"
        minSdk = 26
        targetSdk = 34
        versionCode = 2
        versionName = "1.1"
        // Real-phones-only app: ship arm64-v8a native libs only (Battles's
        // call 2026-10-11; x86_64 emulators are not supported).
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    signingConfigs {
        // Release signing reads the keystore from CI secrets (see the release
        // job in .github/workflows/native-android.yml). Local release builds
        // without KEYSTORE_PATH skip signing (unsigned release APK).
        create("release") {
            val ksPath = System.getenv("KEYSTORE_PATH")
            if (!ksPath.isNullOrBlank()) {
                storeFile = file(ksPath)
                storePassword = System.getenv("KEYSTORE_PASSWORD")
                keyAlias = System.getenv("KEY_ALIAS")
                keyPassword = System.getenv("KEY_PASSWORD")
            }
        }
    }

    buildTypes {
        release {
            if (!System.getenv("KEYSTORE_PATH").isNullOrBlank()) {
                signingConfig = signingConfigs.getByName("release")
            }
            // R8 is on for release; keep-rules for the ORT JNI bridge live in
            // proguard-rules.pro. Debug stays unshrunk.
            isMinifyEnabled = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    sourceSets {
        named("main") {
            assets.srcDir(layout.buildDirectory.dir("generated/assets"))
        }
    }
}

tasks.named("preBuild") {
    dependsOn(copyModels, copyModelExtras)
}

dependencies {
    // On-device inference. Version pinned; check Maven Central for updates.
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.22.0")
    // Localhost HTTP server for the external-agent API (VisionService).
    implementation("org.nanohttpd:nanohttpd:2.3.1")
    // FileProvider (camera capture without CAMERA permission) and
    // Activity Result APIs.
    implementation("androidx.core:core:1.13.1")
    implementation("androidx.activity:activity:1.9.2")
}
