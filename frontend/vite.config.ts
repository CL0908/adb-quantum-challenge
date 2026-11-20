import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Relative base: the built site is plain static files in ../web, served by Vercel next to the api/ function.
export default defineConfig({
  base: './',
  plugins: [react()],
  build: { outDir: '../web', emptyOutDir: true },
})
