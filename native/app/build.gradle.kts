plugins {
    id("com.android.application")
    kotlin("android")
}

// The .onnx models, synset.txt and test-bus.jpg live at the repo root (shared
// with the Termux Python server). They are copied into a generated assets
// directory at build time so the APK stays self-contained without duplicating
// ~54 MB of binaries in git.
val copyModelAssets = tasks.register<Copy>("copyModelAssets") {
    from(
        rootProject.file("../yolov8n.onnx"),
        rootProject.file("../yolov8n-seg.onnx"),
        rootProject.file("../yolov8n-pose.onnx"),
        rootProject.file("../mobilenetv2-12.onnx"),
    ) {
        into("models")
    }
    from(
        rootProject.file("../synset.txt"),
        rootProject.file("../test-bus.jpg"),
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
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
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
            assets.srcDir(copyModelAssets.map { it.destinationDir })
        }
    }
}

tasks.named("preBuild") {
    dependsOn(copyModelAssets)
}

dependencies {
    // On-device inference. Version pinned; check Maven Central for updates.
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.22.0")
}
