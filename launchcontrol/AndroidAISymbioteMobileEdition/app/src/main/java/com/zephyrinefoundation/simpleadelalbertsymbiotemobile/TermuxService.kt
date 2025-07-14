package com.zephyrinefoundation.simpleadelalbertsymbiotemobile

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.localbroadcastmanager.content.LocalBroadcastManager
// This is the ONLY place we should reference the bootstrap code.
import com.termux.app.TermuxInstaller // Using an alias for clarity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class TermuxService : Service() {

    private val job = SupervisorJob()
    private val serviceScope = CoroutineScope(Dispatchers.IO + job)
    private var serverProcess: Process? = null

    private val notificationManager by lazy {
        getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
    }

    // This companion object ONLY holds constants for this class.
    // The native function declaration has been completely removed.
    companion object {
        const val ACTION_LOG_BROADCAST = "com.zephyrinefoundation.simpleadelalbertsymbiotemobile.LOG_BROADCAST"
        const val EXTRA_LOG_LINE = "EXTRA_LOG_LINE"
        private const val NOTIFICATION_CHANNEL_ID = "TermuxServiceChannel"
        private const val NOTIFICATION_ID = 1
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val notification = createNotification("Core service is starting...")
        startForeground(NOTIFICATION_ID, notification)

        serviceScope.launch {
            try {
                broadcastLog("Service received start command.")
                bootstrapTermux()
                startServer()
            } catch (e: Exception) {
                Log.e("TermuxService", "Fatal error during service operation.", e)
                broadcastLog("FATAL ERROR: ${e.message}")
                stopSelf()
            }
        }
        return START_STICKY
    }

    override fun onDestroy() {
        super.onDestroy()
        broadcastLog("Service stopping gracefully...")
        Log.d("TermuxService", "Service is being destroyed.")
        serverProcess?.destroy() // Use destroy() for graceful shutdown (SIGTERM)
        serverProcess = null
        job.cancel()
        Log.d("TermuxService", "All service coroutines cancelled.")
    }

    private fun bootstrapTermux() {
        val homeDir = filesDir
        val bootstrapFlag = File(homeDir, ".bootstrapped")

        if (bootstrapFlag.exists()) {
            broadcastLog("Termux environment already set up.")
            return
        }

        broadcastLog("Termux bootstrap not found. Starting initial setup...")
        val architecture = Build.SUPPORTED_ABIS[0]
        val bootstrapZipFileName = when (architecture) {
            "arm64-v8a" -> "bootstrap-aarch64.zip"
            "x86_64" -> "bootstrap-x86_64.zip"
            else -> {
                val errorMessage = "Unsupported CPU architecture: $architecture."
                Log.e("TermuxService", errorMessage)
                broadcastLog("FATAL: $errorMessage")
                throw IllegalStateException(errorMessage)
            }
        }
        broadcastLog("Detected architecture: $architecture. Using $bootstrapZipFileName.")

        try {
            val bootstrapZipFile = File(cacheDir, bootstrapZipFileName)
            assets.open(bootstrapZipFileName).use { assetInputStream ->
                FileOutputStream(bootstrapZipFile).use { fileOutputStream ->
                    assetInputStream.copyTo(fileOutputStream)
                }
            }
            Log.d("TermuxService", "Copied asset to temporary file: ${bootstrapZipFile.absolutePath}")

            broadcastLog("Executing native bootstrap...")
            // Explicitly load the native library before calling the native method.
            // This is more robust than relying on a static initializer.
            TermuxInstaller.loadLibrary()
            // The call now uses the imported bridge class.
            val result = TermuxInstaller.bootstrap(homeDir.absolutePath, bootstrapZipFile.absolutePath)
            bootstrapZipFile.delete()

            if (result == 0) {
                broadcastLog("Native bootstrap completed successfully!")
                assets.open("start_server.sh").use { input ->
                    val scriptFile = File(homeDir, "start_server.sh")
                    FileOutputStream(scriptFile).use { output ->
                        input.copyTo(output)
                    }
                    scriptFile.setExecutable(true, false)
                }
                broadcastLog("Custom server start script installed.")
                bootstrapFlag.createNewFile()
                broadcastLog("Termux bootstrap completed successfully!")
            } else {
                val errorMessage = "FATAL: Native bootstrap failed with exit code $result."
                Log.e("TermuxService", errorMessage)
                broadcastLog(errorMessage)
                throw IOException(errorMessage)
            }
        } catch (e: Exception) {
            Log.e("TermuxService", "Termux bootstrap failed with an exception.", e)
            broadcastLog("ERROR: Termux bootstrap failed: ${e.message}")
            homeDir.deleteRecursively()
            throw e
        }
    }

    private fun startServer() {
        val homeDir = filesDir
        val shellPath = File(homeDir, "bin/sh")

        if (!shellPath.exists() || !shellPath.canExecute()) {
            val errorMessage = "FATAL: Required executable '${shellPath.absolutePath}' not found or not executable. Please clear app data and retry bootstrap."
            Log.e("TermuxService", errorMessage)
            broadcastLog(errorMessage)
            throw IOException(errorMessage)
        }

        val command = listOf(shellPath.absolutePath, "${homeDir.absolutePath}/start_server.sh")
        val processBuilder = ProcessBuilder(command)
            .directory(homeDir)
            .redirectErrorStream(true)

        val env = processBuilder.environment()
        env["HOME"] = homeDir.absolutePath
        env["PREFIX"] = homeDir.absolutePath
        env["LD_LIBRARY_PATH"] = "${homeDir.absolutePath}/lib"
        env["PATH"] = "${homeDir.absolutePath}/bin:${System.getenv("PATH")}"

        val androidSupportLibPath = File(homeDir, "lib/libandroid-support.so").absolutePath
        if (File(androidSupportLibPath).exists()) {
            env["LD_PRELOAD"] = androidSupportLibPath
        }

        broadcastLog("--- Attempting to start server process with command: $command ---")
        updateNotification("Core server is running.")

        try {
            serverProcess = processBuilder.start()
            serverProcess?.inputStream?.bufferedReader()?.forEachLine { line ->
                broadcastLog(line)
            }
            val exitCode = serverProcess?.waitFor() ?: -1
            val exitMessage = "--- Server process exited with code: $exitCode ---"
            Log.d("TermuxService", exitMessage)
            broadcastLog(exitMessage)
            updateNotification("Core server has stopped.")
        } catch (e: Exception) {
            Log.e("TermuxService", "Failed to start or run server process.", e)
            broadcastLog("ERROR: Server process failed: ${e.message}")
            updateNotification("Core server failed to start.")
            throw e
        }
    }

    // --- Helper functions (no changes) ---
    private fun broadcastLog(message: String) {
        Log.i("TermuxCoreLog", message)
        val intent = Intent(ACTION_LOG_BROADCAST).apply {
            putExtra(EXTRA_LOG_LINE, message)
        }
        LocalBroadcastManager.getInstance(this).sendBroadcast(intent)
    }

    override fun onBind(intent: Intent?): IBinder? {
        return null
    }

    private fun createNotificationChannel() {
        val serviceChannel = NotificationChannel(
            NOTIFICATION_CHANNEL_ID,
            "Local Assistant Core Service",
            NotificationManager.IMPORTANCE_LOW
        )
        notificationManager.createNotificationChannel(serviceChannel)
    }

    private fun createNotification(contentText: String): Notification {
        return NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setContentTitle("Local Assistant Core")
            .setContentText(contentText)
            .setSmallIcon(R.drawable.ic_terminal)
            .setOngoing(true)
            .build()
    }

    private fun updateNotification(contentText: String) {
        val notification = createNotification(contentText)
        notificationManager.notify(NOTIFICATION_ID, notification)
    }
}