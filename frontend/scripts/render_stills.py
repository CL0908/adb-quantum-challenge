"""Render the WebGL-free fallback stills (public/hero-terrain.webp, public/problem-terrain.webp) from the live scene.
Serve the built site first (python -m http.server 4174 --directory web), then: python frontend/scripts/render_stills.py
Opens /?still=hero and /?still=problem (src/home/Still.tsx), waits until the scene is ready, captures it on a transparent background."""
import io
import sys
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:4174"
PUBLIC = Path(__file__).resolve().parents[1] / "public"
with sync_playwright() as pw:
    b = pw.chromium.launch(args=["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist"])
    for name in ("hero", "problem"):
        p = b.new_page(viewport={"width": 2400, "height": 1500}, device_scale_factor=1)
        p.goto(f"{BASE}/?still={name}", wait_until="networkidle")
        p.wait_for_selector("#still[data-done='1']", timeout=60000)
        png = p.locator("#still").screenshot(omit_background=True)
        im = Image.open(io.BytesIO(png)).convert("RGBA")
        im.save(PUBLIC / f"{name}-terrain.webp", quality=88, method=6)
        print(name, im.size, (PUBLIC / f"{name}-terrain.webp").stat().st_size // 1024, "KB")
        p.close()
    b.close()
