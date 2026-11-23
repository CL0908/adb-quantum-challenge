"""Encode the rendered loops for the web: public/anim/<name>.mp4 (H.264) and .webm (VP9), plus a poster .webp taken at
the moment the scene shows its result (used as the still for reduced motion). Run after manim, from frontend/."""
import glob
import subprocess
from pathlib import Path

import av
import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
OUT = Path(__file__).resolve().parents[1] / "public" / "anim"
OUT.mkdir(parents=True, exist_ok=True)
POSTER_AT = {"Rdpo": 0.86, "SmallWorld": 0.86, "Twin": 0.9}
for name, at in POSTER_AT.items():
    src = sorted(glob.glob(f"media/videos/nest_scenes/**/{name}.mp4", recursive=True))[-1]
    stem = OUT / name.lower()
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264", "-crf", "27", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", f"{stem}.mp4"], check=True)
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", src, "-c:v", "libvpx-vp9", "-crf", "36", "-b:v", "0", "-an", f"{stem}.webm"], check=True)
    c = av.open(src)
    s = c.streams.video[0]
    c.seek(int(at * float(s.duration * s.time_base) / s.time_base), stream=s)
    for fr in c.decode(s):
        fr.to_image().save(f"{stem}.webp", quality=88)
        break
    print(name, {e: (stem.with_suffix(e)).stat().st_size // 1024 for e in (".mp4", ".webm", ".webp")}, "KB")
