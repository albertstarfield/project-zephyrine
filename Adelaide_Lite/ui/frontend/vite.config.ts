import { defineConfig } from 'vite';
import { VitePWA } from 'vite-plugin-pwa';

export default defineConfig({
  plugins: [
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.ico', 'ProjectZephy023LogoRenewal.png'],
      manifest: {
        name: 'Adelaide Zephyrine Assistant',
        short_name: 'Zephy',
        description: 'OpenIntellegentiaPlatform Agentic Assistant PWA Interface',
        theme_color: '#000000',
        background_color: '#000000',
        display: 'standalone',
        icons: [
          {
            src: 'ProjectZephy023LogoRenewal.png',
            sizes: '192x192',
            type: 'image/png'
          },
          {
            src: 'ProjectZephy023LogoRenewal.png',
            sizes: '512x512',
            type: 'image/png'
          }
        ]
      }
    })
  ]
});
