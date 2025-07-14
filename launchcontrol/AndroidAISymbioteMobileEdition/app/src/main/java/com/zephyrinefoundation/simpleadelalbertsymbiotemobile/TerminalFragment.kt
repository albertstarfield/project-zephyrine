package com.zephyrinefoundation.simpleadelalbertsymbiotemobile

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Build
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.ScrollView
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.localbroadcastmanager.content.LocalBroadcastManager

class TerminalFragment : Fragment() {

    private lateinit var logTextView: TextView
    private lateinit var scrollView: ScrollView

    private val logReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val logLine = intent?.getStringExtra(TermuxService.EXTRA_LOG_LINE)
            if (logLine != null) {
                logTextView.append("\n$logLine")
                // Scroll to the bottom to show the latest log
                scrollView.post { scrollView.fullScroll(View.FOCUS_DOWN) }
            }
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val view = inflater.inflate(R.layout.fragment_terminal, container, false)

        logTextView = view.findViewById(R.id.tvLogOutput)
        scrollView = view.findViewById(R.id.scrollViewContainer)
        val btnStart = view.findViewById<Button>(R.id.btnStartService)
        val btnStop = view.findViewById<Button>(R.id.btnStopService)

        btnStart.setOnClickListener {
            logTextView.text = "Starting service..."
            val serviceIntent = Intent(activity, TermuxService::class.java)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                requireActivity().startForegroundService(serviceIntent)
            } else {
                requireActivity().startService(serviceIntent)
            }
        }

        btnStop.setOnClickListener {
            logTextView.text = "Stopping service..."
            val serviceIntent = Intent(activity, TermuxService::class.java)
            requireActivity().stopService(serviceIntent)
        }

        return view
    }

    override fun onResume() {
        super.onResume()
        // Use LocalBroadcastManager for better security and efficiency
        LocalBroadcastManager.getInstance(requireContext()).registerReceiver(
            logReceiver, IntentFilter(TermuxService.ACTION_LOG_BROADCAST)
        )
    }

    override fun onPause() {
        super.onPause()
        LocalBroadcastManager.getInstance(requireContext()).unregisterReceiver(logReceiver)
    }
}