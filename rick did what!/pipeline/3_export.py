"""
Stage 3: 9:16 Export — v4 (caption_style.json + edit_rules.json spec)

Caption spec (caption_style.json):
  - Font: Impact (Komika Axis not in system — Impact is the meme equivalent)
  - Color: #5FE3AC (mint green), 9px black stroke, lowercase
  - Cap height ≈150px → font_size 195px for Impact
  - Position: y=1229px (64% of 1920), centered horizontally
  - ONE word per caption, duration clamped 120-600ms (default 280ms)
  - Animation-in: 3-frame scale overshoot 70%→108%→100%
  - Animation-out: instant cut

Edit spec (edit_rules.json):
  - Hard cuts only (no transitions)
  - Per-segment camera: sustained push-in (tight/medium) or snap punch
  - Shot distribution: 70% tight, 20% medium, 10% wide; 25% of cuts → snap punch
  - Lateral crop offset ±5% per segment (simulate speaker tracking)
  - Letterbox removal (active band y: 23.4%-72.4% of source height)

Two-stage export to avoid FFmpeg's simultaneous-decoder-thread limit:
  Stage 1: camera moves → lossless H.264 intermediate
  Stage 2: Pillow caption PNGs applied in batches of 50 → lossless intermediate
  Stage 3: final HEVC 10Mbps encode

No hook text shown in video. Creative filenames. manifest.json for audit compatibility.
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
VIDEO_BITRATE    = "10M"
AUDIO_BITRATE    = "192k"
TARGET_W         = 1080
TARGET_H         = 1920
CAPTION_BATCH_SZ = 50        # max PNG overlays per FFmpeg pass

# ── Caption style (caption_style.json) ───────────────────────────────────────
CAPTION_COLOR     = (95, 227, 172)     # #5FE3AC mint green
CAPTION_OUTLINE   = (0, 0, 0)
CAPTION_STROKE_PX = 9
CAPTION_FONT_SIZE = 195               # ~150px cap height for Impact
CAPTION_POP_70    = int(195 * 0.70)   # 136px — frame 0 (scale-in)
CAPTION_POP_108   = int(195 * 1.08)   # 210px — frame 1 (overshoot)
CAPTION_Y_PX      = 1229             # 64% of 1920
CAPTION_MIN_MS    = 120
CAPTION_MAX_MS    = 600
CAPTION_DEFAULT_MS = 280

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
    """Render one caption word: lowercase, mint green #5FE3AC, Impact, fixed y."""
    text = word.lower().strip()
    if not text:
        return
    img  = Image.new("RGBA", (TARGET_W, TARGET_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _load_font(size)

    bbox   = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    if text_w > TARGET_W - 60:
        font   = _load_font(max(60, size - 30))
        bbox   = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]

    x = (TARGET_W - text_w) // 2
    _outline_text(draw, x, CAPTION_Y_PX, text, font, CAPTION_COLOR)
    img.save(out_path, "PNG")


def build_caption_pngs(
    words_data: list[dict],
    clip_start: float,
    clip_end:   float,
    png_dir:    str,
    clip_id:    str,
) -> list[tuple[str, float, float]]:
    """
    One word per caption with 3-frame overshoot animation.
    Returns sorted list of (png_path, t_start, t_end).
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

    # Pre-render unique (word, size) combinations — avoids duplicate work
    render_cache: dict[tuple[str, int], str] = {}

    def get_png(word: str, size: int, idx: int) -> str:
        key = (word.lower().strip(), size)
        if key not in render_cache:
            path = os.path.join(png_dir, f"{clip_id}_w{idx:04d}_s{size}.png")
            render_word_png(word, path, size)
            render_cache[key] = path
        return render_cache[key]

    entries = []
    for idx, w in enumerate(clip_words):
        raw_ms = (w["end"] - w["start"]) * 1000
        dur_ms = CAPTION_DEFAULT_MS if raw_ms < 50 else max(CAPTION_MIN_MS, min(CAPTION_MAX_MS, raw_ms))

        t_start = w["start"]
        t_end   = t_start + dur_ms / 1000.0
        f0_end  = t_start + 0.033
        f1_end  = t_start + 0.067
        word    = w["word"]

        # Frame 0 — 70% (scale in)
        entries.append((get_png(word, CAPTION_POP_70,    idx, 0), t_start,           min(f0_end, t_end)))
        # Frame 1 — 108% (overshoot)
        if t_end > f0_end:
            entries.append((get_png(word, CAPTION_POP_108,   idx, 1), max(f0_end, t_start + 0.001), min(f1_end, t_end)))
        # Frame 2+ — 100% (normal)
        if t_end > f1_end:
            entries.append((get_png(word, CAPTION_FONT_SIZE, idx, 2), max(f1_end, t_start + 0.002), t_end))

    return entries


# ── Scene detection ───────────────────────────────────────────────────────────

def detect_scene_cuts(video_path: str) -> list[float]:
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


# ── Camera moves ──────────────────────────────────────────────────────────────

def _choose_shot_type(seg_idx: int) -> str:
    rng = random.Random(seg_idx * 7919)
    if rng.random() < 0.25:
        return "snap"
    r2 = rng.random()
    if r2 < 0.70:
        return "tight"
    elif r2 < 0.90:
        return "medium"
    return "wide"


def _zoompan(shot: str, seg_dur: float, crop_w: int, crop_h: int) -> str:
    n  = max(1, int(seg_dur * 30))
    cx = f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={crop_w}x{crop_h}"
    if shot == "tight":
        return f"zoompan=z='min(zoom+{0.10/n:.7f},1.10)':d=1{cx}"
    elif shot == "medium":
        return f"zoompan=z='min(zoom+{0.05/n:.7f},1.05)':d=1{cx}"
    elif shot == "snap":
        return f"zoompan=z='if(lte(on,2),1.0,1.30)':d=1{cx}"
    else:
        return f"zoompan=z='1.0':d=1{cx}"


def build_camera_filter(src_w: int, src_h: int,
                         cut_times: list[float], clip_dur: float) -> str:
    """
    Build filter_complex for camera moves only (no captions).
    Output label: [vbase]
    """
    ar     = 9 / 16
    crop_w = int(src_h * ar)
    crop_h = src_h
    if crop_w > src_w:
        crop_w = src_w
        crop_h = int(src_w / ar)
    cx = (src_w - crop_w) // 2
    cy = (src_h - crop_h) // 2

    # Letterbox removal (spec: active band 23.4%-72.4%)
    lb_top = int(src_h * 0.234)
    lb_bot = int(src_h * 0.724)
    if lb_bot - lb_top >= crop_h:
        cy = lb_top + (lb_bot - lb_top - crop_h) // 2

    zoom_w = int(TARGET_W * 1.05)
    zoom_h = int(TARGET_H * 1.05)
    grade  = (f"scale={zoom_w}:{zoom_h}:flags=lanczos,"
              f"crop={TARGET_W}:{TARGET_H},"
              f"eq=contrast=1.2:saturation=1.6:brightness=0.01,"
              f"vignette=PI/5")

    bounds = [0.0] + list(cut_times) + [clip_dur]
    segs   = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

    fc_parts = []
    labels   = []
    for i, (t_s, t_e) in enumerate(segs):
        dur = t_e - t_s
        if dur < 0.033:
            continue
        shot    = _choose_shot_type(i)
        zp      = _zoompan(shot, dur, crop_w, crop_h)
        # Lateral speaker-tracking offset: -1 / 0 / +1 cycling
        off     = int(crop_w * 0.05) * ((i % 3) - 1)
        seg_cx  = max(0, min(src_w - crop_w, cx + off))
        label   = f"seg{i}"
        fc_parts.append(
            f"[0:v]trim=start={t_s:.3f}:end={t_e:.3f},"
            f"setpts=PTS-STARTPTS,"
            f"crop={crop_w}:{crop_h}:{seg_cx}:{cy},"
            f"{zp},{grade}[{label}]"
        )
        labels.append(label)

    concat = "".join(f"[{l}]" for l in labels)
    fc_parts.append(f"{concat}concat=n={len(labels)}:v=1:a=0[vbase]")
    return ";".join(fc_parts)


# ── Caption overlay (chunked) ─────────────────────────────────────────────────

def apply_caption_batch(
    in_path:  str,
    entries:  list[tuple[str, float, float]],
    out_path: str,
    lossless: bool = True,
) -> None:
    """Apply a batch of ≤CAPTION_BATCH_SZ caption PNG overlays to in_path → out_path."""
    cmd = ["ffmpeg", "-y", "-i", in_path]
    for png_path, _, _ in entries:
        cmd += ["-i", png_path]

    fc_parts = []
    prev     = "0:v"
    for i, (_, t_start, t_end) in enumerate(entries):
        label = f"vc{i}"
        fc_parts.append(
            f"[{prev}][{i + 1}:v]overlay=0:0:"
            f"enable='between(t\\,{t_start:.3f}\\,{t_end:.3f})'[{label}]"
        )
        prev = label

    last = fc_parts[-1]
    fc_parts[-1] = last[:last.rfind("[")] + "[out]"

    if lossless:
        vcodec = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "0"]
    else:
        vcodec = ["-c:v", "hevc_videotoolbox", "-b:v", VIDEO_BITRATE, "-tag:v", "hvc1"]

    cmd += [
        "-filter_complex", ";".join(fc_parts),
        "-map", "[out]", "-map", "0:a",
        *vcodec,
        "-c:a", "copy",
        "-pix_fmt", "yuv420p",
        out_path, "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True)


# ── Slug ─────────────────────────────────────────────────────────────────────

def make_slug(hook_text: str) -> str:
    name = re.sub(r'[^\w\s]', '', hook_text)
    name = name.lower().strip()
    name = re.sub(r'\s+', '-', name)
    return re.sub(r'-+', '-', name).strip('-')[:40] or "clip"


# ── Main export ───────────────────────────────────────────────────────────────

def export_clip(
    video_path:   str,
    start:        float,
    end:          float,
    clip_id:      str,
    hook_text:    str,
    words_data:   list[dict],
    out_dir:      str,
    pipeline_dir: str,
) -> str:
    slug     = make_slug(hook_text)
    out_path = os.path.join(out_dir, f"{slug}.mp4")
    if os.path.exists(out_path):
        print(f"  [skip] {slug}.mp4 already exported")
        return out_path

    png_dir = os.path.join(pipeline_dir, "caption_pngs", clip_id)
    os.makedirs(png_dir, exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix=f"rdw_{clip_id}_")

    try:
        raw = os.path.join(tmp_dir, "raw.mp4")

        # ── Step 1: stream-copy segment ───────────────────────────────────────
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(start), "-to", str(end),
            "-i", video_path,
            "-c", "copy", raw, "-loglevel", "error",
        ], check=True)

        cut_times = detect_scene_cuts(raw)
        clip_dur  = get_clip_duration(raw)
        w, h      = get_dimensions(raw)
        print(f"  Cuts: {len(cut_times)} → {[f'{t:.1f}s' for t in cut_times]}")

        # ── Step 2: camera moves → lossless intermediate ──────────────────────
        cam_out = os.path.join(tmp_dir, "camera.mp4")
        cam_fc  = build_camera_filter(w, h, cut_times, clip_dur)
        subprocess.run([
            "ffmpeg", "-y", "-i", raw,
            "-filter_complex", cam_fc,
            "-map", "[vbase]", "-map", "0:a",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            cam_out, "-loglevel", "error",
        ], check=True)

        # ── Step 3: render caption PNGs ───────────────────────────────────────
        caption_entries = build_caption_pngs(words_data, start, end, png_dir, clip_id)
        n_words = len([e for e in caption_entries if e[0].endswith("_s136.png") or "_s195.png" in e[0] or "_s210.png" in e[0]])
        print(f"  Caption frames: {len(caption_entries)} ({len(caption_entries)//3 if caption_entries else 0} words × 3)")

        # ── Step 4: apply captions in batches of CAPTION_BATCH_SZ ────────────
        current = cam_out
        batches = [caption_entries[i:i + CAPTION_BATCH_SZ]
                   for i in range(0, len(caption_entries), CAPTION_BATCH_SZ)]

        for b_idx, batch in enumerate(batches):
            batch_out = os.path.join(tmp_dir, f"cap_{b_idx}.mp4")
            apply_caption_batch(current, batch, batch_out, lossless=True)
            current = batch_out
            print(f"  Caption batch {b_idx + 1}/{len(batches)} applied")

        # ── Step 5: final HEVC encode ─────────────────────────────────────────
        subprocess.run([
            "ffmpeg", "-y", "-i", current,
            "-c:v", "hevc_videotoolbox",
            "-b:v", VIDEO_BITRATE,
            "-tag:v", "hvc1",
            "-c:a", "aac",
            "-b:a", AUDIO_BITRATE,
            "-movflags", "+faststart",
            "-metadata", f"title={clip_id}",
            out_path, "-loglevel", "error",
        ], check=True)

        size_mb = os.path.getsize(out_path) / 1_000_000
        print(f"  Exported: {slug}.mp4  [{start:.0f}s–{end:.0f}s]  {size_mb:.1f} MB")
        print(f"    Caption: #5FE3AC Impact 1-word, 64% from top, pop-in animation")
        print(f"    Camera : {len(cut_times)} reframes, push-in/punch, lateral tracking")

    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return out_path


def run(candidates_json: str, out_dir: str) -> list[str]:
    with open(candidates_json) as f:
        data = json.load(f)

    episode      = data["episode"]
    clips        = data["clips"]
    pipeline_dir = str(Path(candidates_json).parent)

    words_path = os.path.join(pipeline_dir, f"{episode}_words.json")
    if os.path.exists(words_path):
        with open(words_path) as f:
            words_data = json.load(f)
        print(f"[3_export] Word timestamps: {len(words_data)} words")
    else:
        transcript_path = os.path.join(pipeline_dir, f"{episode}_transcript.json")
        if os.path.exists(transcript_path):
            with open(transcript_path) as f:
                segs = json.load(f)
            words_data = []
            for seg in segs:
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
        else:
            words_data = []
            print("[3_export] WARNING: no word timestamps")

    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[3_export] Episode : {episode}")
    print(f"[3_export] Clips   : {len(clips)}")
    print(f"[3_export] Caption : #5FE3AC Impact, 1-word, 3-frame pop-in, 64% from top")
    print(f"[3_export] Camera  : scene-cut push-in/punch, lateral reframe, no hook overlay")
    print(f"[3_export] Export  : 2-stage (camera → lossless, captions in batches of {CAPTION_BATCH_SZ})")

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
            )
            exported.append(path)
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
