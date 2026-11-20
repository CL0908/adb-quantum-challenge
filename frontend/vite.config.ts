import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { existsSync } from 'node:fs'
import { resolve } from 'node:path'

// Multi-page build: the home page plus one folder per extra page (solution/, dashboard/), each built to web/<name>/index.html.
// A page is included only once its index.html exists, so pages can land independently.
const here = (p: string) => resolve(import.meta.dirname, p)
const input: Record<string, string> = { main: here('index.html') }
for (const page of ['solution', 'dashboard']) {
  if (existsSync(here(`${page}/index.html`))) input[page] = here(`${page}/index.html`)
}

// Relative base: the built site is plain static files in ../web, served by Vercel next to the api/ function.
export default defineConfig({
  base: './',
  plugins: [react()],
  build: { outDir: '../web', emptyOutDir: true, rollupOptions: { input } },
})
