"""
Stage 3: 9:16 Export — v4 (caption_style.json + edit_rules.json spec)

Caption spec (caption_style.json):
  - Font: Impact (Komika Axis fallback — not on system)
  - Color: #5FE3AC (mint green), 9px black stroke, lowercase
  - Cap height: 150px → font_size ≈ 195px for Impact
  - Position: y=1229px (64% of 1920), center horizontal
  - One word per caption, duration clamped 120-600ms (default 280ms)
  - Animation-in: 3-frame scale overshoot 70%→108%→100%
  - Animation-out: instant cut

Edit spec (edit_rules.json):
  - Hard cuts only, no transitions
  - Per-segment camera moves: sustained push-in (tight/medium) or snap punch
  - Shot distribution: 70% tight, 20% medium, 10% wide; 25% of cuts → snap punch
  - Lateral crop offset alternates ±5% per segment to simulate speaker framing
  - Letterbox removal (active band y: 23.4%-72.4% of source)

Filenames: creative slug from hook_text (e.g. "jerry-is-doomed.mp4")
No hook text overlay in video — slug used only for filename.
manifest.json written to clips/ for audit stage compatibility.
"""

import argparse
import json
import os
import random
import re
import subprocess
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── Export settings ───────────────────────────────────────────────────────────
VIDEO_BITRATE = "10M"
AUDIO_BITRATE = "192k"
TARGET_W      = 1080
TARGET_H      = 1920

# ── Caption style (caption_style.json) ───────────────────────────────────────
CAPTION_COLOR     = (95, 227, 172)     # #5FE3AC mint green
CAPTION_OUTLINE   = (0, 0, 0)
CAPTION_STROKE_PX = 9                 # uniform hard stroke
CAPTION_FONT_SIZE = 195               # ≈150px cap height for Impact
CAPTION_POP_70    = int(195 * 0.70)   # 136px — frame 0 (scale in)
CAPTION_POP_108   = int(195 * 1.08)   # 210px — frame 1 (overshoot)
CAPTION_Y_PX      = 1229             # fixed y — 64% of 1920
CAPTION_MIN_MS    = 120
CAPTION_MAX_MS    = 600
CAPTION_DEFAULT_MS = 280

# ── Font paths ────────────────────────────────────────────────────────────────
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
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


def _outline_text(draw, x, y, text, font, fill, border: int = CAPTION_STROKE_PX):
    for dx, dy in [(-border, 0), (border, 0), (0, -border), (0, border),
                   (-border, -border), (-border, border), (border, -border), (border, border)]:
        draw.text((x + dx, y + dy), text, font=font, fill=(*CAPTION_OUTLINE, 255))
    draw.text((x, y), text, font=font, fill=(*fill, 255))


def render_word_png(word: str, out_path: str, size: int = CAPTION_FONT_SIZE) -> None:
    """Render a single caption word: lowercase, mint green, Impact, fixed y=1229."""
    text = word.lower().strip()
    if not text:
        return
    img  = Image.new("RGBA", (TARGET_W, TARGET_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _load_font(size)

    bbox   = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    # Safety clamp if text exceeds frame width
    if text_w > TARGET_W - 60:
        font   = _load_font(max(60, size - 30))
        bbox   = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]

    x = (TARGET_W - text_w) // 2
    y = CAPTION_Y_PX
    _outline_text(draw, x, y, text, font, CAPTION_COLOR)
    img.save(out_path, "PNG")


def build_caption_pngs(
    words_data: list[dict],
    clip_start: float,
    clip_end:   float,
    png_dir:    str,
    clip_id:    str,
) -> list[tuple[str, float, float]]:
    """
    One word per caption, 3-frame scale overshoot animation on entry.
    Returns list of (png_path, t_start, t_end) sorted by time.
    """
    clip_words = [
        {
            "start": round(w["start"] - clip_start, 3),
            "end":   round(w["end"]   - clip_start, 3),
            "word":  w["word"],
        }
        for w in words_data
        if w["start"] >= clip_start - 0.05 and w["end"] <= clip_end + 0.05
           and w["word"].strip()
    ]
    if not clip_words:
        return []

    entries = []
    for idx, w in enumerate(clip_words):
        word    = w["word"]
        t_start = w["start"]
        raw_ms  = (w["end"] - w["start"]) * 1000

        # Clamp duration to spec range
        if raw_ms < 50:
            dur_ms = CAPTION_DEFAULT_MS
        else:
            dur_ms = max(CAPTION_MIN_MS, min(CAPTION_MAX_MS, raw_ms))

        t_end  = t_start + dur_ms / 1000.0
        f0_end = t_start + 0.033   # frame 0 end (1 frame @30fps)
        f1_end = t_start + 0.067   # frame 1 end (2 frames @30fps)

        base = os.path.join(png_dir, f"{clip_id}_w{idx:04d}")

        # Frame 0 — 70% size (scale-in)
        f0 = base + "_f0.png"
        render_word_png(word, f0, size=CAPTION_POP_70)
        entries.append((f0, t_start, min(f0_end, t_end)))

        # Frame 1 — 108% size (overshoot)
        if t_end > f0_end:
            f1 = base + "_f1.png"
            render_word_png(word, f1, size=CAPTION_POP_108)
            entries.append((f1, f0_end, min(f1_end, t_end)))

        # Frame 2+ — 100% normal size
        if t_end > f1_end:
            f2 = base + "_f2.png"
            render_word_png(word, f2, size=CAPTION_FONT_SIZE)
            entries.append((f2, f1_end, t_end))

    return entries


# ── Scene detection ───────────────────────────────────────────────────────────

def detect_scene_cuts(video_path: str) -> list[float]:
    """Return cut timestamps (0-relative seconds) via PySceneDetect."""
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
        video = open_video(video_path)
        sm    = SceneManager()
        sm.add_detector(ContentDetector(threshold=27))
        sm.detect_scenes(video, show_progress=False)
        scenes = sm.get_scene_list()
        return [s[0].get_seconds() for s in scenes[1:]]
    except Exception as e:
        print(f"  [scene detect] skipped: {e}")
        return []


def get_clip_duration(video_path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 40.0


def get_dimensions(path: str) -> tuple[int, int]:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0", path,
    ], stderr=subprocess.DEVNULL).decode().strip()
    w, h = out.split(",")
    return int(w), int(h)


# ── Camera moves + filter building ───────────────────────────────────────────

def _choose_shot_type(seg_idx: int) -> str:
    """
    70% tight (push-in 1.0→1.10), 20% medium (1.0→1.05), 10% wide (static).
    25% of cuts override to snap punch (1.0→1.30 instant).
    Deterministic via segment index for reproducibility.
    """
    rng = random.Random(seg_idx * 7919)   # seeded per-segment
    r   = rng.random()
    if r < 0.25:
        return "snap"
    r2 = rng.random()
    if r2 < 0.70:
        return "tight"
    elif r2 < 0.90:
        return "medium"
    return "wide"


def _zoompan_expr(shot_type: str, seg_dur: float, crop_w: int, crop_h: int) -> str:
    n_frames = max(1, int(seg_dur * 30))
    if shot_type == "tight":
        delta = 0.10 / n_frames
        return (f"zoompan=z='min(zoom+{delta:.7f},1.10)':d=1"
                f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={crop_w}x{crop_h}")
    elif shot_type == "medium":
        delta = 0.05 / n_frames
        return (f"zoompan=z='min(zoom+{delta:.7f},1.05)':d=1"
                f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={crop_w}x{crop_h}")
    elif shot_type == "snap":
        return (f"zoompan=z='if(lte(on,2),1.0,1.30)':d=1"
                f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={crop_w}x{crop_h}")
    else:  # wide — static
        return (f"zoompan=z='1.0':d=1"
                f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={crop_w}x{crop_h}")


def build_video_filter_complex(
    src_w: int,
    src_h: int,
    cut_times: list[float],
    clip_duration: float,
    caption_entries: list[tuple[str, float, float]],
) -> tuple[str, list[int]]:
    """
    Build the full filter_complex string and return (filter_str, caption_input_indices).

    Layout:
      Input 0: raw clip
      Inputs 1..N: caption PNGs (no hook PNG)

    Filter stages per segment:
      trim → setpts → crop (with lateral offset) → zoompan → scale → crop → eq → vignette

    Concat all segments → [vbase]
    Chain caption overlays on [vbase] → [out]
    """
    # 9:16 crop geometry
    ar     = 9 / 16
    crop_w = int(src_h * ar)
    crop_h = src_h
    if crop_w > src_w:
        crop_w = src_w
        crop_h = int(src_w / ar)
    center_x = (src_w - crop_w) // 2
    center_y = (src_h - crop_h) // 2

    # Letterbox adjustment (spec: active band 23.4%–72.4%)
    lb_top    = int(src_h * 0.234)
    lb_bottom = int(src_h * 0.724)
    active_h  = lb_bottom - lb_top
    if active_h >= crop_h:
        center_y = lb_top + (active_h - crop_h) // 2

    # Output scale (1.05× overshoot then crop back)
    zoom_w = int(TARGET_W * 1.05)
    zoom_h = int(TARGET_H * 1.05)

    grade = f"scale={zoom_w}:{zoom_h}:flags=lanczos,crop={TARGET_W}:{TARGET_H},eq=contrast=1.2:saturation=1.6:brightness=0.01,vignette=PI/5"

    # Segments from cut times
    boundaries = [0.0] + list(cut_times) + [clip_duration]
    segments   = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    n_segs     = len(segments)

    fc_parts = []
    seg_labels = []

    for i, (t_s, t_e) in enumerate(segments):
        seg_dur  = t_e - t_s
        if seg_dur < 0.033:          # skip sub-frame segments
            continue

        shot     = _choose_shot_type(i)
        zp_expr  = _zoompan_expr(shot, seg_dur, crop_w, crop_h)

        # Lateral crop offset — alternate ±5% per segment
        direction   = (i % 3) - 1   # -1, 0, +1 cycling
        lateral_px  = int(crop_w * 0.05)
        seg_crop_x  = max(0, min(src_w - crop_w, center_x + direction * lateral_px))

        label = f"seg{i}"
        fc_parts.append(
            f"[0:v]trim=start={t_s:.3f}:end={t_e:.3f},"
            f"setpts=PTS-STARTPTS,"
            f"crop={crop_w}:{crop_h}:{seg_crop_x}:{center_y},"
            f"{zp_expr},"
            f"{grade}[{label}]"
        )
        seg_labels.append(label)

    # Concat all segments
    concat_in = "".join(f"[{l}]" for l in seg_labels)
    fc_parts.append(f"{concat_in}concat=n={len(seg_labels)}:v=1:a=0[vbase]")

    # Caption overlays — input indices start at 1 (no hook PNG)
    prev = "vbase"
    caption_input_indices = list(range(1, 1 + len(caption_entries)))

    for i, (_, t_start, t_end) in enumerate(caption_entries):
        inp   = caption_input_indices[i]
        label = f"vc{i}"
        fc_parts.append(
            f"[{prev}][{inp}:v]overlay=0:0:"
            f"enable='between(t\\,{t_start:.3f}\\,{t_end:.3f})'[{label}]"
        )
        prev = label

    # Final label
    last = fc_parts[-1]
    fc_parts[-1] = last[:last.rfind("[")] + "[out]"

    return ";".join(fc_parts), caption_input_indices


# ── Filename slug ─────────────────────────────────────────────────────────────

def make_slug(hook_text: str) -> str:
    """'JERRY IS DOOMED 😱' → 'jerry-is-doomed'"""
    name = re.sub(r'[^\w\s]', '', hook_text)
    name = name.lower().strip()
    name = re.sub(r'\s+', '-', name)
    name = re.sub(r'-+', '-', name).strip('-')
    return name[:40] or "clip"


# ── Main export function ──────────────────────────────────────────────────────

def export_clip(
    video_path:   str,
    start:        float,
    end:          float,
    clip_id:      str,
    hook_text:    str,
    words_data:   list[dict],
    out_dir:      str,
    pipeline_dir: str,
    score:        float = 0.0,
) -> str:
    slug     = make_slug(hook_text)
    out_path = os.path.join(out_dir, f"{slug}.mp4")

    if os.path.exists(out_path):
        print(f"  [skip] {slug}.mp4 already exported")
        return out_path

    png_dir = os.path.join(pipeline_dir, "caption_pngs", clip_id)
    os.makedirs(png_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        raw = f.name

    try:
        # Step 1: stream-copy segment (t=0 relative in output)
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(start), "-to", str(end),
            "-i", video_path,
            "-c", "copy", raw,
            "-loglevel", "error",
        ], check=True)

        # Step 2: detect scene cuts
        cut_times = detect_scene_cuts(raw)
        clip_dur  = get_clip_duration(raw)
        print(f"  Cuts: {len(cut_times)} → {[f'{t:.1f}s' for t in cut_times]}")

        # Step 3: build caption PNGs (one word, 3-frame overshoot)
        caption_entries = build_caption_pngs(words_data, start, end, png_dir, clip_id)
        print(f"  Caption frames: {len(caption_entries)} ({len([e for e in caption_entries if '_f0' in e[0]])} words × 3)")

        # Step 4: source dimensions
        w, h = get_dimensions(raw)

        # Step 5: build filter_complex
        fc_str, cap_indices = build_video_filter_complex(
            w, h, cut_times, clip_dur, caption_entries
        )

        # Step 6: assemble FFmpeg command
        # Input 0: raw  |  Inputs 1+: caption PNGs (no hook PNG)
        cmd = ["ffmpeg", "-y", "-i", raw]
        for png_path, _, _ in caption_entries:
            cmd += ["-i", png_path]
        cmd += [
            "-filter_complex", fc_str,
            "-map", "[out]",
            "-map", "0:a",
            "-c:v", "hevc_videotoolbox",
            "-b:v", VIDEO_BITRATE,
            "-tag:v", "hvc1",
            "-c:a", "aac",
            "-b:a", AUDIO_BITRATE,
            "-movflags", "+faststart",
            "-metadata", f"title={clip_id}",   # preserve technical ID in metadata
            out_path,
            "-loglevel", "error",
        ]

        subprocess.run(cmd, check=True)

        size_mb = os.path.getsize(out_path) / 1_000_000
        print(f"  Exported: {slug}.mp4  [{start:.0f}s–{end:.0f}s]  {size_mb:.1f} MB")

    finally:
        if os.path.exists(raw):
            os.unlink(raw)

    return out_path


# ── run() ─────────────────────────────────────────────────────────────────────

def run(candidates_json: str, out_dir: str) -> list[str]:
    with open(candidates_json) as f:
        data = json.load(f)

    episode      = data["episode"]
    clips        = data["clips"]
    pipeline_dir = str(Path(candidates_json).parent)

    # Load word timestamps
    words_path = os.path.join(pipeline_dir, f"{episode}_words.json")
    if os.path.exists(words_path):
        with open(words_path) as f:
            words_data = json.load(f)
        print(f"[3_export] Word timestamps: {len(words_data)} words")
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
            print("[3_export] WARNING: no word timestamps — captions absent")

    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[3_export] Episode  : {episode}")
    print(f"[3_export] Clips    : {len(clips)}")
    print(f"[3_export] Caption  : #5FE3AC Impact, 1-word, 64% from top, pop-in animation")
    print(f"[3_export] Camera   : scene-cut push-in/punch, lateral reframe, no hook overlay")

    manifest_path = os.path.join(out_dir, "manifest.json")
    manifest      = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    exported = []
    for clip in clips:
        try:
            slug = make_slug(clip.get("hook_text", "clip"))
            path = export_clip(
                video_path=clip["video_path"],
                start=clip["start"],
                end=clip["end"],
                clip_id=clip["clip_id"],
                hook_text=clip.get("hook_text", "clip"),
                words_data=words_data,
                out_dir=out_dir,
                pipeline_dir=pipeline_dir,
                score=clip.get("score", 0.0),
            )
            exported.append(path)

            # Write manifest entry
            manifest[slug] = {
                "clip_id":   clip["clip_id"],
                "episode":   episode,
                "score":     clip.get("score", 0.0),
                "hook_text": clip.get("hook_text", ""),
            }
        except Exception as e:
            print(f"  [error] {clip.get('clip_id', '?')}: {e}")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[3_export] manifest → {manifest_path}")
    print(f"[3_export] Done: {len(exported)}/{len(clips)} clips → {out_dir}/")
    return exported


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Export 9:16 Shorts")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output",     default="clips")
    args = parser.parse_args()
    run(args.candidates, args.output)
