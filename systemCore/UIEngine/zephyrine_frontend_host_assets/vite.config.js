// externalAnalyzer/frontend-face-zephyrine/vite.config.js
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa'; //

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      injectRegister: 'auto',
      workbox: {
        globPatterns: ['**/*.{js,css,html,ico,png,svg,jpg,jpeg,woff,woff2,ttf,otf}'],
        // NEW: Add runtimeCaching rules to explicitly bypass Service Worker for API calls
        runtimeCaching: [
          {
            urlPattern: ({ url }) => url.pathname === '/primedready', // Target the specific /primedready path
            handler: 'NetworkOnly', // Go straight to network, do not cache
            options: {
              // No additional options needed for NetworkOnly strategy to simply fetch.
            }
          },
          {
            urlPattern: ({ url }) => url.pathname.startsWith('/api'), // Also apply to your /api routes
            handler: 'NetworkOnly', // Ensure API calls also bypass SW cache
            options: {
              // No additional options needed.
            }
          },
        ]
      },
      includeAssets: ['/img/AdelaideEntity.png', '/img/ProjectZephy023LogoRenewal.png', '/fonts/*.ttf', '/fonts/*.otf'],
      manifest: {
        name: 'Project Zephyrine',
        short_name: 'Zephyrine',
        description: 'Suspiciously an AI',
        theme_color: '#242424',
        background_color: '#1a1a1a',
        display: 'standalone',
        scope: '/',
        start_url: '/',
        icons: [
          {
            src: '/img/AdelaideEntity_192.png',
            sizes: '192x192',
            type: 'image/png',
            purpose: 'any',
          },
          {
            src: '/img/AdelaideEntity_512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'any',
          },
           {
            src: '/img/AdelaideEntity_maskable.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'maskable'
          }
        ]
      },
      devOptions: {
        enabled: true, // Keep PWA features enabled in development
      },
    }),
  ],
  server: { //
    proxy: {
      '/api': {
        target: 'http://localhost:3001',
        changeOrigin: true,
        secure: false
      },
      '/primedready': 'http://localhost:3001', //
      '/ZephyCortexConfig': 'http://localhost:3001'
    }
  }
});