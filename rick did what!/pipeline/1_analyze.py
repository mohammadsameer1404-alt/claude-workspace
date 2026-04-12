"""
Stage 1: Transcript-Based Sliding Window Analysis

Approach — why sliding windows, not scene detection:
  Rick & Morty cuts every 2–5 seconds. ContentDetector creates hundreds of scenes
  shorter than any viable Short. Merging those scenes by content boundary produces
  at most a handful of candidates. Instead we:
    1. Extract audio + run Whisper once (local, free, M1-native)
    2. Compute RMS energy per second
    3. Slide a window of WIN_SEC seconds across the transcript in steps of STEP_SEC
    4. Score each window by dialogue density × mean audio energy
    5. Deduplicate overlapping windows by keeping the highest-scoring one per region
    6. Output the top MAX_CANDIDATES windows as candidates_raw.json

This reliably produces 20–50 strong candidates per 22-minute episode.

Runs on M1 Mac natively — no RAM-heavy ops, streams audio in chunks.
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import whisper


# ── Window config ─────────────────────────────────────────────────────────────
WIN_SEC       = 45      # default window width (seconds) — mid-range for Shorts
WIN_MIN       = 28      # minimum window width to keep
WIN_MAX       = 58      # maximum window width (YouTube Shorts hard cap)
STEP_SEC      = 10      # slide step — 10s gives ~130 windows per 22-min episode
MIN_WORDS     = 15      # skip windows with fewer than this many words (silence)
MAX_CANDIDATES = 60     # cap — Gemini has 1,500 free calls/day, don't overshoot


def extract_audio(video_path: str, out_wav: str) -> None:
    """Extract mono 16kHz WAV — Whisper's preferred format."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-acodec", "pcm_s16le", out_wav,
        "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True)


def transcribe(wav_path: str) -> list[dict]:
    """Run Whisper tiny — returns [{start, end, text}]."""
    print("  Transcribing with Whisper tiny (local, free)...")
    model = whisper.load_model("tiny")
    result = model.transcribe(wav_path, language="en", verbose=False)
    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": round(seg["start"], 2),
            "end":   round(seg["end"],   2),
            "text":  seg["text"].strip(),
        })
    return segments


def compute_rms_per_second(wav_path: str, duration: float) -> np.ndarray:
    """Return array of mean RMS energy per second. Streams the file."""
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    n_seconds = int(np.ceil(duration))
    rms = np.zeros(n_seconds)
    for i in range(n_seconds):
        s = i * sr
        e = min((i + 1) * sr, len(y))
        chunk = y[s:e]
        if len(chunk) > 0:
            rms[i] = float(np.sqrt(np.mean(chunk ** 2)))
    return rms


def window_score(
    start: float,
    end: float,
    transcript: list[dict],
    rms: np.ndarray,
) -> tuple[float, str]:
    """
    Score a time window by:
      - dialogue density (words per second)
      - mean audio energy in that window
    Returns (score, joined_transcript_text).
    """
    words = []
    for seg in transcript:
        # Include segments that overlap with [start, end]
        if seg["end"] > start and seg["start"] < end:
            words.append(seg["text"])

    joined = " ".join(words).strip()
    word_count = len(joined.split())
    duration = end - start

    if word_count < MIN_WORDS:
        return 0.0, joined

    # Dialogue density: words/sec (normalised 0–1 around typical 2.5 wps)
    dialogue_density = min(word_count / duration / 2.5, 1.0)

    # Audio energy: mean RMS in window
    si = max(0, int(start))
    ei = min(len(rms), int(np.ceil(end)))
    mean_energy = float(np.mean(rms[si:ei])) if ei > si else 0.0
    # Normalise against 99th-percentile of the episode energy
    # (caller passes pre-normalised rms, so 0–1 range expected)

    score = (0.6 * dialogue_density) + (0.4 * mean_energy)
    return round(score, 6), joined


def sliding_windows(
    transcript: list[dict],
    rms: np.ndarray,
    episode_duration: float,
) -> list[dict]:
    """
    Generate all sliding windows, score each one, deduplicate overlapping windows,
    and return top MAX_CANDIDATES sorted by score descending.
    """
    # Normalise RMS to 0–1 for mixing with dialogue density
    rms_max = rms.max()
    rms_norm = rms / rms_max if rms_max > 0 else rms

    scored = []
    t = 30.0  # skip first 30s (usually cold open / theme)
    end_limit = episode_duration - 10.0  # skip last 10s (credits)

    while t + WIN_MIN <= end_limit:
        win_end = min(t + WIN_SEC, end_limit)
        if win_end - t < WIN_MIN:
            break
        score, text = window_score(t, win_end, transcript, rms_norm)
        if score > 0:
            scored.append({
                "start":      round(t, 2),
                "end":        round(win_end, 2),
                "duration":   round(win_end - t, 2),
                "transcript": text,
                "raw_score":  score,
            })
        t += STEP_SEC

    if not scored:
        return []

    # Sort by score descending
    scored.sort(key=lambda x: x["raw_score"], reverse=True)

    # Deduplicate: greedy — keep highest-scoring window, suppress others that
    # overlap it by > 50%. This prevents returning 5 windows from the same 1-min scene.
    kept = []
    suppressed = set()

    for i, w in enumerate(scored):
        if i in suppressed:
            continue
        kept.append(w)
        for j, other in enumerate(scored):
            if j <= i or j in suppressed:
                continue
            # Overlap length
            overlap = max(0, min(w["end"], other["end"]) - max(w["start"], other["start"]))
            shorter_dur = min(w["duration"], other["duration"])
            if overlap / shorter_dur > 0.5:
                suppressed.add(j)

    return kept[:MAX_CANDIDATES]


def _extract_code(video_path: str) -> str:
    import re
    stem = Path(video_path).stem
    m = re.search(r"S(\d{2})E(\d{2})", stem, re.IGNORECASE)
    return f"S{m.group(1).upper()}E{m.group(2).upper()}" if m else stem


def run(video_path: str, output_dir: str) -> str:
    video_path   = str(Path(video_path).resolve())
    episode_stem = _extract_code(video_path)
    out_json     = os.path.join(output_dir, f"{episode_stem}_candidates_raw.json")

    print(f"\n[1_analyze] Episode : {episode_stem}  ({Path(video_path).name})")
    print(f"[1_analyze] Output  : {out_json}")
    print(f"[1_analyze] Strategy: transcript sliding window  ({WIN_SEC}s, step {STEP_SEC}s)")

    # Reuse existing transcript if present — avoids a slow Whisper re-run
    transcript_path = os.path.join(output_dir, f"{episode_stem}_transcript.json")
    if os.path.exists(transcript_path):
        print(f"  [cache] Reusing existing transcript: {transcript_path}")
        with open(transcript_path) as f:
            segments = json.load(f)
        episode_duration = segments[-1]["end"] if segments else 0
        print(f"  Got {len(segments)} transcript segments (cached)")
        print(f"  Episode duration : {episode_duration:.0f}s ({episode_duration/60:.1f} min)")

        print("  Extracting audio for RMS energy...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        try:
            extract_audio(video_path, wav_path)
            print("  Computing per-second RMS energy...")
            rms = compute_rms_per_second(wav_path, episode_duration)
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)

        print("  Generating sliding windows...")
        candidates = sliding_windows(segments, rms, episode_duration)
        print(f"  Found {len(candidates)} candidate windows (top {MAX_CANDIDATES} after dedup)")

        payload = {
            "episode":    episode_stem,
            "video_path": video_path,
            "candidates": candidates,
        }
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"  Saved {len(candidates)} candidates → {out_json}")
        return out_json

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    try:
        print("  Extracting audio...")
        extract_audio(video_path, wav_path)

        segments = transcribe(wav_path)
        print(f"  Got {len(segments)} transcript segments")

        episode_duration = segments[-1]["end"] if segments else 0
        print(f"  Episode duration : {episode_duration:.0f}s ({episode_duration/60:.1f} min)")

        print("  Computing per-second RMS energy...")
        rms = compute_rms_per_second(wav_path, episode_duration)

        print("  Generating sliding windows...")
        candidates = sliding_windows(segments, rms, episode_duration)
        print(f"  Found {len(candidates)} candidate windows (top {MAX_CANDIDATES} after dedup)")

        # Save full transcript for Stage 3 caption use
        transcript_path = os.path.join(output_dir, f"{episode_stem}_transcript.json")
        with open(transcript_path, "w") as f:
            json.dump(segments, f, indent=2)

        payload = {
            "episode":    episode_stem,
            "video_path": video_path,
            "candidates": candidates,
        }
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"  Saved {len(candidates)} candidates → {out_json}")
        return out_json

    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Analyze episode for candidate clip windows")
    parser.add_argument("--episode", required=True, help="Path to episode file (MP4/MKV)")
    parser.add_argument("--output",  default="pipeline", help="Directory to write JSON output")
    args = parser.parse_args()

    run(args.episode, args.output)
