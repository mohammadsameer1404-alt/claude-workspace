"""
Stage 3: Premium High-Resolution 9:16 Clip Export

Quality:
  - Stream-copy segment first (zero quality loss), then single re-encode pass
  - H.265 via M1 VideoToolbox @ 10 Mbps + 192k AAC
  - 1080×1920 (9:16) — crop centred, scale with lanczos
  - Hook title: Pillow-rendered PNG overlaid for first 2.5s (upper third, white + black outline)
  - Captions: neon green (#39FF14) burnt-in via ASS when libass is available;
              falls back to clean export (YouTube auto-captions cover the rest)
  - faststart for instant YouTube playback

Credit usage: ZERO — all FFmpeg + Pillow, no API calls.
"""

import argparse
import json
import os
import subprocess
import tempfile
from PIL import Image, ImageDraw, ImageFont

# ── Export settings ───────────────────────────────────────────────────────────
VIDEO_BITRATE = "10M"
AUDIO_BITRATE = "192k"
TARGET_W      = 1080
TARGET_H      = 1920

# Neon green (RGB) for captions
CAPTION_COLOR = (57, 255, 20)   # #39FF14


# ── Pillow hook title ─────────────────────────────────────────────────────────

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Black.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Black.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def render_hook_png(hook_text: str, out_path: str) -> None:
    """
    Render hook title as a transparent PNG sized to the output frame.
    White bold text with thick black outline, centred in the upper third.
    """
    img  = Image.new("RGBA", (TARGET_W, TARGET_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _load_font(72)

    bbox   = draw.textbbox((0, 0), hook_text, font=font)
    text_w = bbox[2] - bbox[0]
    x      = (TARGET_W - text_w) // 2
    y      = int(TARGET_H * 0.18)

    # Black outline (4 px)
    for dx in range(-4, 5):
        for dy in range(-4, 5):
            if dx or dy:
                draw.text((x + dx, y + dy), hook_text, font=font, fill=(0, 0, 0, 255))
    # White fill
    draw.text((x, y), hook_text, font=font, fill=(255, 255, 255, 255))

    img.save(out_path, "PNG")


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_dimensions(path: str) -> tuple[int, int]:
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0", path,
    ], stderr=subprocess.DEVNULL).decode().strip()
    w, h = out.split(",")
    return int(w), int(h)


def crop_scale_filter(src_w: int, src_h: int) -> str:
    """Centre-crop to 9:16 then scale to 1080×1920."""
    ar     = 9 / 16
    crop_w = int(src_h * ar)
    crop_h = src_h
    if crop_w > src_w:
        crop_w = src_w
        crop_h = int(src_w / ar)
    x = (src_w - crop_w) // 2
    y = (src_h - crop_h) // 2
    return f"crop={crop_w}:{crop_h}:{x}:{y},scale={TARGET_W}:{TARGET_H}:flags=lanczos"


# ── Main export ───────────────────────────────────────────────────────────────

def export_clip(
    video_path: str,
    start: float,
    end: float,
    clip_id: str,
    hook_text: str,
    out_dir: str,
) -> str:
    out_path = os.path.join(out_dir, f"{clip_id}.mp4")
    if os.path.exists(out_path):
        print(f"  [skip] {clip_id}.mp4 already exported")
        return out_path

    with tempfile.NamedTemporaryFile(suffix=".mp4",  delete=False) as f:
        raw = f.name
    with tempfile.NamedTemporaryFile(suffix=".png",  delete=False) as f:
        hook_png = f.name

    try:
        # ── Step 1: stream-copy segment ───────────────────────────────────────
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(start), "-to", str(end),
            "-i", video_path,
            "-c", "copy", raw,
            "-loglevel", "error",
        ], check=True)

        # ── Step 2: render hook PNG ───────────────────────────────────────────
        render_hook_png(hook_text, hook_png)

        # ── Step 3: get source dimensions ─────────────────────────────────────
        w, h    = get_dimensions(raw)
        vf_base = crop_scale_filter(w, h)

        # ── Step 4: build filter_complex ──────────────────────────────────────
        # Overlay the hook PNG for the first 2.5s.
        # \, is FFmpeg's escaped comma in a filter option value — parsed as literal ","
        # so  lt(t\,2.5)  →  lt(t,2.5)  inside the timeline expression.
        show_until = min(2.5, (end - start) * 0.25)
        fc = (
            f"[0:v]{vf_base}[base];"
            f"[base][1:v]overlay=0:0:enable=lt(t\\,{show_until})[out]"
        )

        subprocess.run([
            "ffmpeg", "-y",
            "-i", raw,
            "-i", hook_png,
            "-filter_complex", fc,
            "-map", "[out]",
            "-map", "0:a",
            "-c:v", "hevc_videotoolbox",
            "-b:v", VIDEO_BITRATE,
            "-tag:v", "hvc1",
            "-c:a", "aac",
            "-b:a", AUDIO_BITRATE,
            "-movflags", "+faststart",
            "-metadata", f"title={clip_id}",
            out_path,
            "-loglevel", "error",
        ], check=True)

        size_mb = os.path.getsize(out_path) / 1_000_000
        print(f"  Exported: {clip_id}.mp4  [{start:.0f}s–{end:.0f}s]  {size_mb:.1f} MB")
        print(f"    Hook : \"{hook_text}\"")

    finally:
        for p in [raw, hook_png]:
            if os.path.exists(p):
                os.unlink(p)

    return out_path


def run(candidates_json: str, out_dir: str) -> list[str]:
    with open(candidates_json) as f:
        data = json.load(f)

    episode = data["episode"]
    clips   = data["clips"]

    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[3_export] Episode  : {episode}")
    print(f"[3_export] Winners  : {len(clips)}")
    print(f"[3_export] Quality  : {TARGET_W}x{TARGET_H} @ {VIDEO_BITRATE} H.265 (M1 HW)")
    print(f"[3_export] Audio    : AAC {AUDIO_BITRATE}")
    print(f"[3_export] Hook     : Pillow PNG overlay (first 2.5s)")

    exported = []
    for clip in clips:
        try:
            path = export_clip(
                video_path=clip["video_path"],
                start=clip["start"],
                end=clip["end"],
                clip_id=clip["clip_id"],
                hook_text=clip.get("hook_text", "Rick and Morty 😭"),
                out_dir=out_dir,
            )
            exported.append(path)
        except Exception as e:
            print(f"  [error] {clip.get('clip_id', '?')}: {e}")

    print(f"\n  Done: {len(exported)}/{len(clips)} clips → {out_dir}/")
    return exported


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Export 9:16 Shorts")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output",     default="clips")
    args = parser.parse_args()
    run(args.candidates, args.output)
