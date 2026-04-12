"""
Stage 3: Premium 9:16 Clip Export with Burned-In Captions

Changes from v1:
  - Real caption burning via Pillow PNG overlays (no libass needed)
  - Caption style: neon green (#39FF14), 70% from top, Arial Black 68px, 4px black outline
  - Word-pop: active word shown in WHITE, rest of line in neon green
  - Hook text: white, centered, 15% from top, 2.5s — now at 90px and properly positioned
  - No more filter_complex dependency — uses -vf chain for captions
  - H.265 M1 hardware, 10Mbps + 192k AAC

Note: libass is NOT required. All captions are pre-rendered as transparent Pillow PNGs
and chained via FFmpeg overlay filter with time-based enable expressions.
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

# ── Caption style ─────────────────────────────────────────────────────────────
CAPTION_COLOR      = (57, 255, 20)      # neon green #39FF14
CAPTION_COLOR_POP  = (255, 255, 255)    # white — active word pop
CAPTION_OUTLINE    = (0, 0, 0)          # black outline
CAPTION_FONT_SIZE  = 68
CAPTION_Y_RATIO    = 0.70               # 70% from top
CAPTION_MAX_CHARS  = 35                 # max chars per caption line

# ── Hook text style ───────────────────────────────────────────────────────────
HOOK_FONT_SIZE     = 90
HOOK_Y_RATIO       = 0.15              # 15% from top
HOOK_DURATION      = 2.5              # seconds

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
    for dx, dy in [(-border,0),(border,0),(0,-border),(0,border),
                   (-border,-border),(-border,border),(border,-border),(border,border)]:
        draw.text((x + dx, y + dy), text, font=font, fill=(*outline, 255))
    draw.text((x, y), text, font=font, fill=(*fill, 255))


def render_hook_png(hook_text: str, out_path: str) -> None:
    """Render hook title: white text, 90px, centered at 15% from top."""
    img  = Image.new("RGBA", (TARGET_W, TARGET_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _load_font(HOOK_FONT_SIZE)

    bbox   = draw.textbbox((0, 0), hook_text, font=font)
    text_w = bbox[2] - bbox[0]
    x      = (TARGET_W - text_w) // 2
    y      = int(TARGET_H * HOOK_Y_RATIO)

    _outline_text(draw, x, y, hook_text, font, (255, 255, 255), border=5)
    img.save(out_path, "PNG")


def _group_words_into_lines(clip_words: list[dict]) -> list[list[dict]]:
    """Group clip-relative words into caption lines (max CAPTION_MAX_CHARS chars each)."""
    lines   = []
    current = []
    cur_len = 0

    for w in clip_words:
        word_text = w["word"]
        add_len   = len(word_text) + (1 if current else 0)  # +1 for space
        if current and cur_len + add_len > CAPTION_MAX_CHARS:
            lines.append(current)
            current = [w]
            cur_len = len(word_text)
        else:
            current.append(w)
            cur_len += add_len

    if current:
        lines.append(current)
    return lines


def _measure_word_x(draw, line_words: list[dict], word_idx: int,
                    start_x: int, font) -> int:
    """Return pixel x-offset of word[word_idx] within the line."""
    prefix = ""
    for i in range(word_idx):
        prefix += line_words[i]["word"] + " "
    if not prefix:
        return start_x
    bbox = draw.textbbox((0, 0), prefix, font=font)
    return start_x + (bbox[2] - bbox[0])


def render_caption_line_png(line_words: list[dict], out_path: str,
                             pop_word_idx: int | None = None) -> None:
    """
    Render one caption line as transparent PNG.
    If pop_word_idx is None: entire line in neon green (base layer).
    If pop_word_idx is set: just the pop word in WHITE (pop layer, overlaid on base).
    """
    img  = Image.new("RGBA", (TARGET_W, TARGET_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _load_font(CAPTION_FONT_SIZE)

    full_text = " ".join(w["word"] for w in line_words)
    bbox      = draw.textbbox((0, 0), full_text, font=font)
    text_w    = bbox[2] - bbox[0]
    start_x   = (TARGET_W - text_w) // 2
    y         = int(TARGET_H * CAPTION_Y_RATIO)

    if pop_word_idx is None:
        # Base layer: entire line in neon green
        _outline_text(draw, start_x, y, full_text, font, CAPTION_COLOR)
    else:
        # Pop layer: only the active word in white (no background, no green text)
        wx    = _measure_word_x(draw, line_words, pop_word_idx, start_x, font)
        wtext = line_words[pop_word_idx]["word"]
        _outline_text(draw, wx, y, wtext, font, CAPTION_COLOR_POP)

    img.save(out_path, "PNG")


def build_caption_pngs(
    words_data: list[dict],
    clip_start: float,
    clip_end: float,
    png_dir: str,
    clip_id: str,
) -> list[tuple[str, float, float]]:
    """
    Pre-render all caption PNGs for a clip.
    Returns list of (png_path, enable_start, enable_end) sorted by time.
    """
    # Filter and retime words to clip-relative timestamps
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

    lines   = _group_words_into_lines(clip_words)
    entries = []   # (png_path, start, end)

    for line_idx, line_words in enumerate(lines):
        if not line_words:
            continue

        line_start = line_words[0]["start"]
        line_end   = line_words[-1]["end"]

        # Base layer: neon green line (visible for full line duration)
        base_path = os.path.join(png_dir, f"{clip_id}_L{line_idx:03d}_base.png")
        render_caption_line_png(line_words, base_path, pop_word_idx=None)
        entries.append((base_path, line_start, line_end))

        # Pop layers: one per word (white active word overlaid on top of green)
        for word_idx, word in enumerate(line_words):
            pop_path = os.path.join(png_dir, f"{clip_id}_L{line_idx:03d}_W{word_idx:03d}.png")
            render_caption_line_png(line_words, pop_path, pop_word_idx=word_idx)
            entries.append((pop_path, word["start"], word["end"]))

    # Sort by start time so overlay order is chronological
    entries.sort(key=lambda x: x[1])
    return entries


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
        # Step 1: stream-copy segment
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(start), "-to", str(end),
            "-i", video_path,
            "-c", "copy", raw,
            "-loglevel", "error",
        ], check=True)

        # Step 2: render hook PNG
        render_hook_png(hook_text, hook_png)

        # Step 3: build caption PNGs
        caption_entries = build_caption_pngs(words_data, start, end, png_dir, clip_id)
        print(f"  Rendered {len(caption_entries)} caption overlays")

        # Step 4: get source dimensions
        w, h    = get_dimensions(raw)
        vf_base = crop_scale_filter(w, h)

        # Step 5: build FFmpeg filter_complex with all overlays
        # Input 0: raw video, Input 1: hook PNG, Inputs 2+: caption PNGs
        show_hook_until = min(HOOK_DURATION, (end - start) * 0.25)

        all_inputs = [raw, hook_png] + [e[0] for e in caption_entries]
        n_extra    = len(caption_entries)  # number of caption inputs (index 2 onward)

        # Build filter_complex
        fc_parts = [f"[0:v]{vf_base}[vbase]"]
        prev     = "vbase"

        # Hook overlay (input index 1)
        fc_parts.append(
            f"[{prev}][1:v]overlay=0:0:enable='lt(t\\,{show_hook_until:.2f})'[vh]"
        )
        prev = "vh"

        # Caption overlays (input indices 2, 3, ...)
        for i, (_, t_start, t_end) in enumerate(caption_entries):
            inp_idx = i + 2
            label   = f"vc{i}"
            fc_parts.append(
                f"[{prev}][{inp_idx}:v]overlay=0:0:"
                f"enable='between(t\\,{t_start:.3f}\\,{t_end:.3f})'[{label}]"
            )
            prev = label

        # Rename final label to [out]
        fc_parts[-1] = fc_parts[-1].rsplit(f"[{prev}]", 1)[0] + "[out]"
        if not caption_entries:
            # No captions — rename [vh] to [out]
            fc_parts[-1] = fc_parts[-1].replace("[vh]", "[out]")

        filter_complex = ";".join(fc_parts)

        # Build FFmpeg command
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
        print(f"    Captions: neon green, 70% down, word-pop ✓")

    finally:
        if os.path.exists(raw):
            os.unlink(raw)
        # Keep caption PNGs for debugging; remove if you prefer clean runs
        # shutil.rmtree(png_dir, ignore_errors=True)

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
        # Fallback: estimate from transcript
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
    print(f"[3_export] Captions : neon green #{'{:02X}{:02X}{:02X}'.format(*CAPTION_COLOR)}, "
          f"{int(CAPTION_Y_RATIO*100)}% from top, word-pop")

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
