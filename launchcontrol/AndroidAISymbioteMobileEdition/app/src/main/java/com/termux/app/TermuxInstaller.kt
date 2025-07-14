package com.termux.app

/**
 * FINAL CORRECTED JNI BRIDGE
 * The native function in `libtermux-bootstrap.so` is actually expecting to be called
 * from a class named `com.termux.app.TermuxInstaller`.
 *
 * This class provides that exact structure.
 */
object TermuxInstaller {
    private var libraryLoaded = false

    /**
     * Loads the native library. This must be called before any native methods.
     * It's synchronized to prevent race conditions in multi-threaded environments.
     */
    @Synchronized
    fun loadLibrary() {
        if (libraryLoaded) return
        System.loadLibrary("termux-bootstrap")
        libraryLoaded = true
    }

    /**
     * The native JNI function to perform the bootstrap. Its real C name is
     * `Java_com_termux_app_TermuxInstaller_bootstrap`. This Kotlin declaration
     * now matches it perfectly.
     */
    @JvmStatic
    external fun bootstrap(homePath: String, zipPath: String): Int
}