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
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (
            id.includes('/src/components/mesh-topology-diagram.tsx')
            || id.includes('/node_modules/@xyflow/')
          ) {
            return 'dashboard-topology';
          }

          if (
            id.includes('/src/components/chat-markdown-message.tsx')
            || id.includes('/node_modules/react-markdown/')
            || id.includes('/node_modules/remark-gfm/')
            || id.includes('/node_modules/remark-math/')
            || id.includes('/node_modules/rehype-highlight/')
            || id.includes('/node_modules/highlight.js/')
          ) {
            return 'chat-markdown';
          }

          return undefined;
        },
      },
    },
  },
});
