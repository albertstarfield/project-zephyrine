package com.zephyrinefoundation.simpleadelalbertsymbiotemobile

import android.content.Intent
import android.os.Bundle
import android.view.MenuItem
import androidx.appcompat.app.ActionBarDrawerToggle
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.GravityCompat
import androidx.drawerlayout.widget.DrawerLayout
import androidx.fragment.app.Fragment
import com.google.android.material.navigation.NavigationView

class MainActivity : AppCompatActivity(), NavigationView.OnNavigationItemSelectedListener {

    private lateinit var drawerLayout: DrawerLayout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        drawerLayout = findViewById(R.id.drawer_layout)
        val toolbar = findViewById<androidx.appcompat.widget.Toolbar>(R.id.toolbar)
        setSupportActionBar(toolbar) // Set our custom Toolbar as the Activity's action bar

        val navigationView = findViewById<NavigationView>(R.id.nav_view)
        navigationView.setNavigationItemSelectedListener(this)

        // Setup the hamburger icon to open/close the navigation drawer
        val toggle = ActionBarDrawerToggle(this, drawerLayout, toolbar, R.string.open_nav, R.string.close_nav)
        drawerLayout.addDrawerListener(toggle)
        toggle.syncState() // Synchronize the indicator with the drawer's state

        // Load the default fragment (TerminalFragment) when the Activity is first created
        if (savedInstanceState == null) {
            replaceFragment(TerminalFragment())
            navigationView.setCheckedItem(R.id.nav_terminal) // Highlight the current item in the drawer

            // --- AUTO-START THE SERVICE ON APP LAUNCH ---
            val serviceIntent = Intent(this, TermuxService::class.java)
            // Use startForegroundService for Android 8 (Oreo) and above for background tasks
            startForegroundService(serviceIntent)
        }
    }

    // Handles item clicks in the navigation drawer
    override fun onNavigationItemSelected(item: MenuItem): Boolean {
        when (item.itemId) {
            R.id.nav_terminal -> replaceFragment(TerminalFragment())
            R.id.nav_webview -> replaceFragment(WebFragment())
        }
        // Close the drawer after an item is selected
        drawerLayout.closeDrawer(GravityCompat.START)
        return true // Indicate that the item selection was handled
    }

    // Helper function to replace the current fragment in the container
    private fun replaceFragment(fragment: Fragment) {
        supportFragmentManager.beginTransaction()
            .replace(R.id.fragment_container, fragment) // Replace fragment in the fragment_container
            .commit() // Commit the transaction
    }

}