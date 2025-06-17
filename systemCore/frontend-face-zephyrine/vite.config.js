// externalAnalyzer/frontend-face-zephyrine/vite.config.js
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa'; // Import the plugin

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate', // Automatically update the service worker when new content is available
      injectRegister: 'auto', // Let the plugin handle injecting the service worker registration script
      workbox: {
        // Configures Workbox (generates the service worker)
        globPatterns: ['**/*.{js,css,html,ico,png,svg,jpg,jpeg,woff,woff2,ttf,otf}'], // Cache common static assets
        // runtimeCaching: [ // Optional: Add rules for caching API requests if needed later
        //   {
        //     urlPattern: ({ url }) => url.pathname.startsWith('/api'), // Example: Cache API calls
        //     handler: 'NetworkFirst', // Strategy: Try network, fallback to cache
        //     options: {
        //       cacheName: 'api-cache',
        //       expiration: {
        //         maxEntries: 10,
        //         maxAgeSeconds: 60 * 60 * 24 // 1 day
        //       },
        //       cacheableResponse: {
        //         statuses: [0, 200]
        //       }
        //     }
        //   }
        // ]
      },
      includeAssets: ['/img/AdelaideEntity.png', '/img/ProjectZephy023LogoRenewal.png', '/fonts/*.ttf', '/fonts/*.otf'], // Explicitly include important assets from public
      manifest: {
        // --- Customize Your App Manifest ---
        name: 'Project Zephyrine',
        short_name: 'Zephyrine',
        description: 'Allegedly an AI - Adelaide Zephyrine Charlotte',
        theme_color: '#242424', // Match your app's dark theme (adjust if needed)
        background_color: '#1a1a1a', // Background for splash screen
        display: 'standalone', // Run in its own window
        scope: '/',
        start_url: '/', // Start at the root when launched
        icons: [
          // --- IMPORTANT: Create icons of these sizes ---
          // Use AdelaideEntity.png as the base, but generate specific sizes.
          // Place these in `public/icons/` (or adjust path below)
          {
            src: '/img/AdelaideEntity_192.png', // Example path: public/img/AdelaideEntity_192.png
            sizes: '192x192',
            type: 'image/png',
            purpose: 'any', // 'any' or 'maskable' - maskable helps with adaptive icons
          },
          {
            src: '/img/AdelaideEntity_512.png', // Example path: public/img/AdelaideEntity_512.png
            sizes: '512x512',
            type: 'image/png',
            purpose: 'any',
          },
          // Add more sizes if needed (e.g., 144x144 for maskable)
           {
            src: '/img/AdelaideEntity_maskable.png', // Example maskable icon
            sizes: '512x512',
            type: 'image/png',
            purpose: 'maskable'
          }
        ]
      },
      devOptions: {
        enabled: true, // Enable PWA features in development mode
      },
    }),
  ],
  server: { // This 'server' block has been moved here, outside of 'VitePWA' and 'manifest'
    proxy: {
      '/api': 'http://localhost:3001' // Proxy /api requests to your backend server
    }
  }
});