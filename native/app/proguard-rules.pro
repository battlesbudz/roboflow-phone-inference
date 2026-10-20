# ONNX Runtime loads its JNI bridge via System.loadLibrary with reflection;
# keep everything under its package so R8 cannot strip it.
-keep class ai.onnxruntime.** { *; }
-keepclasseswithmembernames class * {
    native <methods>;
}
-dontwarn ai.onnxruntime.**

# androidx.security:security-crypto (Tink) references jsr305/errorprone
# annotations absent from the classpath; silence them so R8 can complete.
-dontwarn javax.annotation.**
-dontwarn com.google.errorprone.**
