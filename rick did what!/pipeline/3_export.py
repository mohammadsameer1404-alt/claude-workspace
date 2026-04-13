"""
Stage 3: 9:16 Export — v4 (caption_style.json + edit_rules.json spec)

Caption spec (caption_style.json):
  - Font: Impact (Komika Axis not on system — Impact is the meme-equivalent)
  - Color: #5FE3AC (mint green), 9px black stroke, lowercase
  - Cap height ≈150px → font_size 195px for Impact
  - Position: y=1229px (64% of 1920), centered horizontally
  - ONE word per caption, duration clamped 120-600ms (default 280ms)
  - Animation-in: 3-frame scale overshoot 70%→108%→100% via drawtext chain
  - Animation-out: instant cut

Edit spec (edit_rules.json):
  - Hard cuts only (no transitions)
  - Per-segment camera: sustained push-in (tight/medium) or snap punch
  - Shot distribution: 70% tight, 20% medium, 10% wide; 25% of cuts → snap punch
  - Lateral crop offset ±5% per segment (simulate speaker tracking)
  - Letterbox removal (active band y: 23.4%-72.4% of source height)

Captions use drawtext filter (no PNG inputs) — avoids OS file descriptor limits.
Filenames: creative slug from hook_text. No hook text shown in video.
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

# ── Export settings ───────────────────────────────────────────────────────────
VIDEO_BITRATE = "10M"
AUDIO_BITRATE = "192k"
TARGET_W      = 1080
TARGET_H      = 1920

# ── Caption style (caption_style.json) ───────────────────────────────────────
CAPTION_COLOR_HEX = "5FE3AC"          # #5FE3AC mint green
CAPTION_STROKE_PX = 9
CAPTION_FONT_SIZE = 195               # ~150px cap height for Impact
CAPTION_POP_70    = int(195 * 0.70)   # 136px frame 0
CAPTION_POP_108   = int(195 * 1.08)   # 210px frame 1
CAPTION_Y_PX      = 1229             # 64% of 1920
CAPTION_MIN_MS    = 120
CAPTION_MAX_MS    = 600
CAPTION_DEFAULT_MS = 280

IMPACT_FONT = "/System/Library/Fonts/Supplemental/Impact.ttf"
FONT_FALLBACKS = [
    IMPACT_FONT,
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]


def _find_font() -> str:
    for p in FONT_FALLBACKS:
        if os.path.exists(p):
            return p
    return ""


def _escape_drawtext(text: str) -> str:
    """Escape text for FFmpeg drawtext filter."""
    text = text.replace("\\", "\\\\")
    text = text.replace("'",  "\u2019")   # replace curly apostrophe to avoid shell issues
    text = text.replace(":",  "\\:")
    text = text.replace("%",  "\\%")
    return text


# ── Caption timing ────────────────────────────────────────────────────────────

def build_caption_words(
    words_data: list[dict],
    clip_start: float,
    clip_end:   float,
) -> list[dict]:
    """
    Return list of {word, t_start, t_end} dicts (clip-relative, clamped duration).
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

    result = []
    for w in clip_words:
        raw_ms = (w["end"] - w["start"]) * 1000
        if raw_ms < 50:
            dur_ms = CAPTION_DEFAULT_MS
        else:
            dur_ms = max(CAPTION_MIN_MS, min(CAPTION_MAX_MS, raw_ms))
        result.append({
            "word":    w["word"],
            "t_start": w["start"],
            "t_end":   w["start"] + dur_ms / 1000.0,
        })
    return result


def build_drawtext_filter(caption_words: list[dict], font_path: str) -> str:
    """
    Build a comma-chained sequence of drawtext filters for the caption words.
    Each word gets 3 drawtext entries (overshoot animation: 70%→108%→100%).
    No file inputs needed — pure FFmpeg filter.
    """
    if not caption_words or not font_path:
        return ""

    parts = []
    fp    = font_path.replace("'", "\\'").replace(":", "\\:")

    for w in caption_words:
        text    = _escape_drawtext(w["word"].lower().strip())
        t_start = w["t_start"]
        t_end   = w["t_end"]
        f0_end  = t_start + 0.033   # frame 0 end
        f1_end  = t_start + 0.067   # frame 1 end

        frames = [
            (CAPTION_POP_70,    t_start,         min(f0_end, t_end)),
            (CAPTION_POP_108,   max(f0_end, t_start + 0.001), min(f1_end, t_end)),
            (CAPTION_FONT_SIZE, max(f1_end, t_start + 0.002), t_end),
        ]

        for size, ts, te in frames:
            if te <= ts + 0.001:
                continue
            parts.append(
                f"drawtext=fontfile='{fp}'"
                f":text='{text}'"
                f":fontsize={size}"
                f":fontcolor=0x{CAPTION_COLOR_HEX}"
                f":borderw={CAPTION_STROKE_PX}"
                f":bordercolor=0x000000"
                f":x=(w-text_w)/2"
                f":y={CAPTION_Y_PX}"
                f":enable='between(t\\,{ts:.3f}\\,{te:.3f})'"
            )

    return ",".join(parts)


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


def _zoompan_expr(shot_type: str, seg_dur: float, crop_w: int, crop_h: int) -> str:
    n  = max(1, int(seg_dur * 30))
    cx = f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={crop_w}x{crop_h}"
    if shot_type == "tight":
        d = 0.10 / n
        return f"zoompan=z='min(zoom+{d:.7f},1.10)':d=1{cx}"
    elif shot_type == "medium":
        d = 0.05 / n
        return f"zoompan=z='min(zoom+{d:.7f},1.05)':d=1{cx}"
    elif shot_type == "snap":
        return f"zoompan=z='if(lte(on,2),1.0,1.30)':d=1{cx}"
    else:
        return f"zoompan=z='1.0':d=1{cx}"


def build_video_filter(
    src_w: int,
    src_h: int,
    cut_times: list[float],
    clip_duration: float,
    drawtext_chain: str,
) -> str:
    """
    Single filter_complex string:
      - Per-segment: trim → setpts → crop (lateral offset) → zoompan → scale → crop → eq → vignette
      - Concat segments → [vbase]
      - Apply drawtext caption chain → [out]
    """
    ar     = 9 / 16
    crop_w = int(src_h * ar)
    crop_h = src_h
    if crop_w > src_w:
        crop_w = src_w
        crop_h = int(src_w / ar)
    center_x = (src_w - crop_w) // 2
    center_y = (src_h - crop_h) // 2

    # Letterbox removal (active band 23.4%-72.4% of source height)
    lb_top   = int(src_h * 0.234)
    lb_bot   = int(src_h * 0.724)
    active_h = lb_bot - lb_top
    if active_h >= crop_h:
        center_y = lb_top + (active_h - crop_h) // 2

    zoom_w = int(TARGET_W * 1.05)
    zoom_h = int(TARGET_H * 1.05)
    grade  = (f"scale={zoom_w}:{zoom_h}:flags=lanczos,"
              f"crop={TARGET_W}:{TARGET_H},"
              f"eq=contrast=1.2:saturation=1.6:brightness=0.01,"
              f"vignette=PI/5")

    boundaries  = [0.0] + list(cut_times) + [clip_duration]
    segments    = [(boundaries[i], boundaries[i + 1])
                   for i in range(len(boundaries) - 1)]

    fc_parts    = []
    seg_labels  = []

    for i, (t_s, t_e) in enumerate(segments):
        seg_dur = t_e - t_s
        if seg_dur < 0.033:
            continue

        shot  = _choose_shot_type(i)
        zp    = _zoompan_expr(shot, seg_dur, crop_w, crop_h)

        # Lateral crop offset — cycle -1/0/+1 per segment
        direction  = (i % 3) - 1
        lateral_px = int(crop_w * 0.05)
        seg_x      = max(0, min(src_w - crop_w, center_x + direction * lateral_px))

        label = f"seg{i}"
        fc_parts.append(
            f"[0:v]trim=start={t_s:.3f}:end={t_e:.3f},"
            f"setpts=PTS-STARTPTS,"
            f"crop={crop_w}:{crop_h}:{seg_x}:{center_y},"
            f"{zp},"
            f"{grade}[{label}]"
        )
        seg_labels.append(label)

    concat_in = "".join(f"[{l}]" for l in seg_labels)
    fc_parts.append(f"{concat_in}concat=n={len(seg_labels)}:v=1:a=0[vbase]")

    # Drawtext caption chain (no extra inputs)
    if drawtext_chain:
        fc_parts.append(f"[vbase]{drawtext_chain}[out]")
    else:
        fc_parts.append("[vbase]null[out]")

    return ";".join(fc_parts)


# ── Filename slug ─────────────────────────────────────────────────────────────

def make_slug(hook_text: str) -> str:
    name = re.sub(r'[^\w\s]', '', hook_text)
    name = name.lower().strip()
    name = re.sub(r'\s+', '-', name)
    name = re.sub(r'-+', '-', name).strip('-')
    return name[:40] or "clip"


# ── Main export ───────────────────────────────────────────────────────────────

def export_clip(
    video_path:   str,
    start:        float,
    end:          float,
    clip_id:      str,
    hook_text:    str,
    words_data:   list[dict],
    out_dir:      str,
) -> str:
    slug     = make_slug(hook_text)
    out_path = os.path.join(out_dir, f"{slug}.mp4")

    if os.path.exists(out_path):
        print(f"  [skip] {slug}.mp4 already exported")
        return out_path

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        raw = f.name

    try:
        # Step 1: stream-copy segment (timestamps reset to 0)
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

        # Step 3: build caption word timings
        caption_words = build_caption_words(words_data, start, end)
        print(f"  Caption words: {len(caption_words)}")

        # Step 4: build drawtext chain
        font_path     = _find_font()
        drawtext_chain = build_drawtext_filter(caption_words, font_path)

        # Step 5: source dimensions + full filter
        w, h = get_dimensions(raw)
        fc   = build_video_filter(w, h, cut_times, clip_dur, drawtext_chain)

        # Step 6: encode (only 1 input — no PNG inputs needed)
        cmd = [
            "ffmpeg", "-y",
            "-i", raw,
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
        ]
        subprocess.run(cmd, check=True)

        size_mb = os.path.getsize(out_path) / 1_000_000
        print(f"  Exported: {slug}.mp4  [{start:.0f}s–{end:.0f}s]  {size_mb:.1f} MB")
        print(f"    Caption : #{CAPTION_COLOR_HEX} Impact, {len(caption_words)} words, 64% from top")
        print(f"    Camera  : {len(cut_times)} cut-point reframes, push-in/punch per segment")

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
            print("[3_export] WARNING: no word timestamps — captions absent")

    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[3_export] Episode : {episode}")
    print(f"[3_export] Clips   : {len(clips)}")
    print(f"[3_export] Caption : #{CAPTION_COLOR_HEX} Impact, 1-word chunks, drawtext (no PNG inputs)")
    print(f"[3_export] Camera  : scene-cut push-in/punch, lateral reframe, no hook overlay")

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
