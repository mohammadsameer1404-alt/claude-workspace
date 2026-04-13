"""
Stage 3: Premium 9:16 Clip Export with Burned-In Captions (v3)

v3 changes:
  - Captions: 72px (was 58px), pop-on-entry animation (96px for 0.15s then 72px)
  - Minimum 0.35s per caption chunk — readable pacing
  - Scene-cut reframe: PySceneDetect finds character cuts, crop X shifts ±4% at each
  - Neon green #39FF14 + black outline, 2 words at a time, 78% from top
  - Color grade: eq=contrast=1.2:saturation=1.6 + vignette, 1.05× zoom
  - No libass — Pillow PNG overlay chain via FFmpeg filter_complex
  - H.265 M1 hardware, 10Mbps + 192k AAC
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── Export settings ───────────────────────────────────────────────────────────
VIDEO_BITRATE = "10M"
AUDIO_BITRATE = "192k"
TARGET_W      = 1080
TARGET_H      = 1920

# ── Caption style (v3) ────────────────────────────────────────────────────────
CAPTION_COLOR        = (57, 255, 20)   # neon green #39FF14
CAPTION_OUTLINE      = (0, 0, 0)       # black outline
CAPTION_FONT_SIZE    = 72              # normal size (was 58)
CAPTION_POP_SIZE     = 96             # pop-entry size (shown for 0.15s on entry)
CAPTION_POP_DURATION = 0.15           # seconds the pop frame is shown
CAPTION_MIN_DURATION = 0.35           # minimum display time per chunk
CAPTION_Y_RATIO      = 0.78           # 78% from top
CAPTION_CHUNK_WORDS  = 2              # words per caption chunk

# ── Hook text style ───────────────────────────────────────────────────────────
HOOK_FONT_SIZE  = 90
HOOK_Y_RATIO    = 0.12
HOOK_DURATION   = 2.5

# ── Font paths ────────────────────────────────────────────────────────────────
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/Library/Fonts/Arial Black.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _outline_text(draw, x, y, text, font, fill, outline=CAPTION_OUTLINE, border=4):
    """Draw text with outline using 8-direction offsets."""
    for dx, dy in [(-border, 0), (border, 0), (0, -border), (0, border),
                   (-border, -border), (-border, border), (border, -border), (border, border)]:
        draw.text((x + dx, y + dy), text, font=font, fill=(*outline, 255))
    draw.text((x, y), text, font=font, fill=(*fill, 255))


def render_hook_png(hook_text: str, out_path: str) -> None:
    """Render hook title: white text, 90px, centered near top."""
    img  = Image.new("RGBA", (TARGET_W, TARGET_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _load_font(HOOK_FONT_SIZE)

    words   = hook_text.split()
    lines   = []
    current = []
    for word in words:
        test = " ".join(current + [word])
        bbox = draw.textbbox((0, 0), test, font=font)
        if current and (bbox[2] - bbox[0]) > TARGET_W - 60:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))

    y      = int(TARGET_H * HOOK_Y_RATIO)
    line_h = draw.textbbox((0, 0), "Ag", font=font)[3] + 8
    for line in lines:
        bbox   = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        x      = (TARGET_W - text_w) // 2
        _outline_text(draw, x, y, line, font, (255, 255, 255), border=5)
        y += line_h

    img.save(out_path, "PNG")


def render_word_chunk_png(words: list[str], out_path: str,
                          size: int = CAPTION_FONT_SIZE) -> None:
    """
    Render 1-2 words as neon green bold text with black outline.
    size controls the font — used for both normal (72px) and pop (96px) frames.
    """
    text = " ".join(words)
    img  = Image.new("RGBA", (TARGET_W, TARGET_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _load_font(size)

    bbox   = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    # Safety clamp — shrink if somehow too wide
    if text_w > TARGET_W - 80:
        font   = _load_font(max(36, size - 16))
        bbox   = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]

    x = (TARGET_W - text_w) // 2
    y = int(TARGET_H * CAPTION_Y_RATIO)
    _outline_text(draw, x, y, text, font, CAPTION_COLOR, border=5)
    img.save(out_path, "PNG")


def build_caption_pngs(
    words_data: list[dict],
    clip_start: float,
    clip_end: float,
    png_dir: str,
    clip_id: str,
) -> list[tuple[str, float, float]]:
    """
    Pre-render caption PNGs — 2 words at a time, neon green, with pop-entry animation.
    Each chunk = pop frame (96px, 0.15s) + normal frame (72px, remainder).
    Returns list of (png_path, enable_start, enable_end).
    """
    clip_words = [
        {
            "start": round(w["start"] - clip_start, 3),
            "end":   round(w["end"]   - clip_start, 3),
            "word":  w["word"],
        }
        for w in words_data
        if w["start"] >= clip_start - 0.1 and w["end"] <= clip_end + 0.1
           and w["word"].strip()
    ]

    if not clip_words:
        return []

    entries = []
    step    = CAPTION_CHUNK_WORDS

    for i in range(0, len(clip_words), step):
        chunk   = clip_words[i : i + step]
        words   = [w["word"] for w in chunk]
        t_start = chunk[0]["start"]
        t_end   = chunk[-1]["end"]

        # Enforce minimum read time
        if t_end - t_start < CAPTION_MIN_DURATION:
            t_end = t_start + CAPTION_MIN_DURATION

        pop_end = min(t_start + CAPTION_POP_DURATION, t_end)

        # Pop frame — bigger font, shown for 0.15s on entry (the "emphasis hit")
        pop_path = os.path.join(png_dir, f"{clip_id}_c{i:04d}_pop.png")
        render_word_chunk_png(words, pop_path, size=CAPTION_POP_SIZE)
        entries.append((pop_path, t_start, pop_end))

        # Normal frame — standard 72px for the rest of the chunk duration
        if t_end > pop_end:
            norm_path = os.path.join(png_dir, f"{clip_id}_c{i:04d}.png")
            render_word_chunk_png(words, norm_path, size=CAPTION_FONT_SIZE)
            entries.append((norm_path, pop_end, t_end))

    return entries


def detect_scene_cuts(video_path: str) -> list[float]:
    """
    Use PySceneDetect to find cut timestamps (seconds, clip-relative) in the clip.
    These mark where characters change on screen — used to reframe the crop.
    Returns [] on failure (graceful fallback to static center crop).
    """
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
        video = open_video(video_path)
        sm    = SceneManager()
        sm.add_detector(ContentDetector(threshold=27))
        sm.detect_scenes(video, show_progress=False)
        scenes = sm.get_scene_list()
        # scenes[0] always starts at 0 — skip it, return starts of subsequent scenes
        return [s[0].get_seconds() for s in scenes[1:]]
    except Exception as e:
        print(f"  [scene detect] skipped: {e}")
        return []


def crop_scale_filter(src_w: int, src_h: int,
                      cut_times: list[float] | None = None) -> str:
    """
    Centre-crop to 9:16, apply dynamic X reframe at scene cuts,
    scale to 1080×1920 with 1.05× zoom-in, then color grade + vignette.
    """
    ar     = 9 / 16
    crop_w = int(src_h * ar)
    crop_h = src_h
    if crop_w > src_w:
        crop_w = src_w
        crop_h = int(src_w / ar)
    x = (src_w - crop_w) // 2
    y = (src_h - crop_h) // 2

    # Build dynamic crop X expression if we have scene cuts
    if cut_times:
        shift  = int(crop_w * 0.04)   # ±4% of crop width per reframe
        x_expr = str(x)               # default = center
        # Build nested if() — earliest cut outermost (checked last, overrides inner)
        for i, cut_t in enumerate(cut_times):
            direction = 1 if i % 2 == 0 else -1
            shifted_x = max(0, min(src_w - crop_w, x + direction * shift))
            # Each new cut wraps previous expression as the else clause
            x_expr = f"if(gte(t\\,{cut_t:.2f})\\,{shifted_x}\\,{x_expr})"
        crop_part = f"crop={crop_w}:{crop_h}:x='{x_expr}':y={y}"
    else:
        crop_part = f"crop={crop_w}:{crop_h}:{x}:{y}"

    zoom_w = int(TARGET_W * 1.05)   # 1134
    zoom_h = int(TARGET_H * 1.05)   # 2016

    return (
        f"{crop_part},"
        f"scale={zoom_w}:{zoom_h}:flags=lanczos,"
        f"crop={TARGET_W}:{TARGET_H},"
        f"eq=contrast=1.2:saturation=1.6:brightness=0.01,"
        f"vignette=PI/5"
    )


def get_dimensions(path: str) -> tuple[int, int]:
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0", path,
    ], stderr=subprocess.DEVNULL).decode().strip()
    w, h = out.split(",")
    return int(w), int(h)


def export_clip(
    video_path: str,
    start: float,
    end: float,
    clip_id: str,
    hook_text: str,
    words_data: list[dict],
    out_dir: str,
    pipeline_dir: str,
) -> str:
    out_path = os.path.join(out_dir, f"{clip_id}.mp4")
    if os.path.exists(out_path):
        print(f"  [skip] {clip_id}.mp4 already exported")
        return out_path

    png_dir = os.path.join(pipeline_dir, "caption_pngs", clip_id)
    os.makedirs(png_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        raw = f.name
    hook_png = os.path.join(png_dir, "_hook.png")

    try:
        # Step 1: stream-copy segment (timestamps reset to 0 in output)
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(start), "-to", str(end),
            "-i", video_path,
            "-c", "copy", raw,
            "-loglevel", "error",
        ], check=True)

        # Step 2: detect scene cuts for reframing
        cut_times = detect_scene_cuts(raw)
        print(f"  Scene cuts: {len(cut_times)} → {[f'{t:.1f}s' for t in cut_times]}")

        # Step 3: render hook PNG
        render_hook_png(hook_text, hook_png)

        # Step 4: build caption PNGs (pop + normal per chunk)
        caption_entries = build_caption_pngs(words_data, start, end, png_dir, clip_id)
        print(f"  Caption frames: {len(caption_entries)} (pop+normal pairs)")

        # Step 5: get source dimensions and build base filter
        w, h    = get_dimensions(raw)
        vf_base = crop_scale_filter(w, h, cut_times)

        # Step 6: build FFmpeg filter_complex
        # Input 0: raw  |  Input 1: hook PNG  |  Inputs 2+: caption PNGs
        show_hook_until = min(HOOK_DURATION, (end - start) * 0.25)
        all_inputs      = [raw, hook_png] + [e[0] for e in caption_entries]

        fc_parts = [f"[0:v]{vf_base}[vbase]"]
        prev     = "vbase"

        # Hook overlay
        fc_parts.append(
            f"[{prev}][1:v]overlay=0:0:enable='lt(t\\,{show_hook_until:.2f})'[vh]"
        )
        prev = "vh"

        # Caption overlays (pop frames and normal frames interleaved)
        for i, (_, t_start, t_end) in enumerate(caption_entries):
            inp_idx = i + 2
            label   = f"vc{i}"
            fc_parts.append(
                f"[{prev}][{inp_idx}:v]overlay=0:0:"
                f"enable='between(t\\,{t_start:.3f}\\,{t_end:.3f})'[{label}]"
            )
            prev = label

        # Rename final label to [out]
        last = fc_parts[-1]
        fc_parts[-1] = last[:last.rfind("[")] + "[out]"

        filter_complex = ";".join(fc_parts)

        # Step 7: encode
        cmd = ["ffmpeg", "-y"]
        for inp in all_inputs:
            cmd += ["-i", inp]
        cmd += [
            "-filter_complex", filter_complex,
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
        ]

        subprocess.run(cmd, check=True)

        size_mb = os.path.getsize(out_path) / 1_000_000
        print(f"  Exported: {clip_id}.mp4  [{start:.0f}s–{end:.0f}s]  {size_mb:.1f} MB")
        print(f"    Hook    : \"{hook_text}\"")
        print(f"    Captions: neon green 72px, pop 96px@0.15s, 78% from top ✓")
        print(f"    Color   : contrast+1.2 saturation+1.6 vignette 1.05× zoom ✓")
        print(f"    Reframe : {len(cut_times)} cut-point shifts applied ✓")

    finally:
        if os.path.exists(raw):
            os.unlink(raw)

    return out_path


def run(candidates_json: str, out_dir: str) -> list[str]:
    with open(candidates_json) as f:
        data = json.load(f)

    episode      = data["episode"]
    clips        = data["clips"]
    pipeline_dir = str(Path(candidates_json).parent)

    # Load word timestamps for captions
    words_path = os.path.join(pipeline_dir, f"{episode}_words.json")
    if os.path.exists(words_path):
        with open(words_path) as f:
            words_data = json.load(f)
        print(f"[3_export] Word timestamps: {len(words_data)} words loaded")
    else:
        transcript_path = os.path.join(pipeline_dir, f"{episode}_transcript.json")
        if os.path.exists(transcript_path):
            with open(transcript_path) as f:
                segments = json.load(f)
            words_data = []
            for seg in segments:
                seg_words = [w for w in seg["text"].split() if w.strip()]
                if not seg_words:
                    continue
                dur = (seg["end"] - seg["start"]) / len(seg_words)
                for i, w in enumerate(seg_words):
                    words_data.append({
                        "start": round(seg["start"] + i * dur, 3),
                        "end":   round(seg["start"] + (i + 1) * dur, 3),
                        "word":  w,
                    })
            print(f"[3_export] Word timestamps: estimated {len(words_data)} from segments")
        else:
            words_data = []
            print("[3_export] WARNING: no word timestamps — captions will be absent")

    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[3_export] Episode  : {episode}")
    print(f"[3_export] Clips    : {len(clips)} (max 3 per episode)")
    print(f"[3_export] Quality  : {TARGET_W}x{TARGET_H} @ {VIDEO_BITRATE} H.265 (M1 HW)")
    print(f"[3_export] Captions : neon green, 2-word chunks, pop animation, 78% from top")
    print(f"[3_export] Effects  : contrast+1.2 saturation+1.6 vignette 1.05× zoom scene-reframe")

    exported = []
    for clip in clips:
        try:
            path = export_clip(
                video_path=clip["video_path"],
                start=clip["start"],
                end=clip["end"],
                clip_id=clip["clip_id"],
                hook_text=clip.get("hook_text", "Rick and Morty 😭"),
                words_data=words_data,
                out_dir=out_dir,
                pipeline_dir=pipeline_dir,
            )
            exported.append(path)
        except Exception as e:
            print(f"  [error] {clip.get('clip_id', '?')}: {e}")

    print(f"\n  Done: {len(exported)}/{len(clips)} clips → {out_dir}/")
    return exported


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Export 9:16 Shorts with captions")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output",     default="clips")
    args = parser.parse_args()
    run(args.candidates, args.output)
