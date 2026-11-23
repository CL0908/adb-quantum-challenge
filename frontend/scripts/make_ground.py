"""Satellite ground texture: NASA Blue Marble Next Generation (public domain) on the DEM's extent, multiplied by a
hillshade computed from the same DEM (light from the north-west). Usage: python frontend/scripts/make_ground.py <bluemarble.png>
Output: public/data/ground.webp, 2048 px wide."""
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "public", "data")
info = json.load(open(os.path.join(DATA, "dem.json"), encoding="utf-8"))
px = np.asarray(Image.open(os.path.join(DATA, "dem.png")).convert("RGB")).astype(np.float64)
h = px[..., 0] * 256 + px[..., 1] - info["offset_m"]
# the imagery crop is fetched once from NASA GIBS (layer BlueMarble_NextGeneration, EPSG:4326, BBOX = dem.json extent, 2960x1620 PNG)
sat = Image.open(sys.argv[1]).convert("RGB")
W, H = sat.size
assert abs(W / H - info["width"] / info["height"]) < 0.01, "imagery and DEM extents differ"
hz = np.asarray(Image.fromarray(h.astype(np.float32), mode="F").resize((W, H), Image.BICUBIC), dtype=np.float64)
lon0, lat0, lon1, lat1 = info["extent"]
mx = 111320 * np.cos(np.radians((lat0 + lat1) / 2)) * (lon1 - lon0) / (W - 1)
my = 111320 * (lat1 - lat0) / (H - 1)
dzdx = np.gradient(hz, axis=1) / mx
dzdy = -np.gradient(hz, axis=0) / my        # rows run north to south
az, alt = np.radians(315), np.radians(40)
slope = np.arctan(np.hypot(dzdx, dzdy) * 1.6)
aspect = np.arctan2(dzdy, -dzdx)
shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
shade = np.clip(shade / np.sin(alt), 0, 1.4)
s = np.asarray(sat, np.float64)
out = s * (0.8 + 0.2 * shade[..., None])  # light baked shading; the real-time key light does the rest
out = np.clip(out * 1.04, 0, 255).astype(np.uint8)
img = Image.fromarray(out)
img = img.resize((2048, round(2048 * img.height / img.width)), Image.LANCZOS)
img.save(os.path.join(DATA, "ground.webp"), quality=88, method=6)
print("saved public/data/ground.webp", img.size)
