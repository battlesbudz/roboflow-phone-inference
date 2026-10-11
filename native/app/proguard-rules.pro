# ONNX Runtime loads its JNI bridge via System.loadLibrary with reflection;
# keep everything under its package so R8 cannot strip it.
-keep class ai.onnxruntime.** { *; }
-keepclasseswithmembernames class * {
    native <methods>;
}
-dontwarn ai.onnxruntime.**
