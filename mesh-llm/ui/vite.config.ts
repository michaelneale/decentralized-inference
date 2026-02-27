import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const apiTarget = process.env.MESH_UI_API_ORIGIN ?? 'http://127.0.0.1:3131';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
});
