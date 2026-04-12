"""
Stage 1: Transcript-Based Sliding Window Analysis

Changes from v1:
  - Whisper now captures word-level timestamps (word_timestamps=True)
  - Saves {episode}_words.json for Stage 3 caption word-pop
  - Added narrative_arc_score() — rewards setup→escalation→payoff pattern
  - New window score: 0.35×dialogue + 0.30×energy + 0.35×arc
  - MAX_CANDIDATES 60→20 (fewer, better candidates fed to Gemini)
  - WIN_SEC 45→40 (tighter default window)
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import whisper


# ── Window config ─────────────────────────────────────────────────────────────
WIN_SEC        = 40     # default window width (seconds)
WIN_MIN        = 28     # minimum window width to keep
WIN_MAX        = 58     # maximum (YouTube Shorts hard cap)
STEP_SEC       = 10     # slide step
MIN_WORDS      = 15     # skip windows with fewer words (silence)
MAX_CANDIDATES = 20     # cap — quality over quantity

# ── Narrative arc signal words ────────────────────────────────────────────────
SETUP_WORDS   = {"what", "why", "how", "where", "when", "who", "listen",
                 "look", "wait", "rick", "morty", "jerry", "beth", "summer",
                 "okay", "alright", "so", "hey"}
PAYOFF_WORDS  = {"exactly", "obviously", "see", "told", "watch", "done",
                 "because", "that's", "there", "wow", "oh", "yes", "no",
                 "right", "boom", "perfect", "never", "always", "literally"}


def extract_audio(video_path: str, out_wav: str) -> None:
    """Extract mono 16kHz WAV — Whisper's preferred format."""
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-acodec", "pcm_s16le", out_wav,
        "-loglevel", "error",
    ], check=True)


def transcribe(wav_path: str) -> tuple[list[dict], list[dict]]:
    """Run Whisper tiny with word-level timestamps.
    Returns (segments, words).
    segments: [{start, end, text}]
    words:    [{start, end, word}]
    """
    print("  Transcribing with Whisper tiny (word timestamps)...")
    model  = whisper.load_model("tiny")
    result = model.transcribe(wav_path, language="en", verbose=False,
                              word_timestamps=True)
    segments = []
    words    = []
    for seg in result["segments"]:
        segments.append({
            "start": round(seg["start"], 2),
            "end":   round(seg["end"],   2),
            "text":  seg["text"].strip(),
        })
        for w in seg.get("words", []):
            word_text = w["word"].strip()
            if word_text:
                words.append({
                    "start": round(float(w["start"]), 3),
                    "end":   round(float(w["end"]),   3),
                    "word":  word_text,
                })
    return segments, words


def estimate_word_timestamps(segments: list[dict]) -> list[dict]:
    """Estimate per-word timing from segment timestamps (fallback)."""
    words = []
    for seg in segments:
        seg_words = [w for w in seg["text"].split() if w.strip()]
        if not seg_words:
            continue
        dur = (seg["end"] - seg["start"]) / len(seg_words)
        for i, w in enumerate(seg_words):
            words.append({
                "start": round(seg["start"] + i * dur, 3),
                "end":   round(seg["start"] + (i + 1) * dur, 3),
                "word":  w,
            })
    return words


def compute_rms_per_second(wav_path: str, duration: float) -> np.ndarray:
    """Return per-second RMS energy array."""
    y, sr  = librosa.load(wav_path, sr=16000, mono=True)
    n_sec  = int(np.ceil(duration))
    rms    = np.zeros(n_sec)
    for i in range(n_sec):
        s = i * sr
        e = min((i + 1) * sr, len(y))
        if e > s:
            rms[i] = float(np.sqrt(np.mean(y[s:e] ** 2)))
    return rms


def narrative_arc_score(
    start: float,
    end: float,
    transcript: list[dict],
    rms: np.ndarray,
) -> float:
    """
    Score how well a window follows setup → escalation → payoff.
    Returns 0.0–1.0.
    """
    duration = end - start
    third    = duration / 3.0

    # Partition transcript into thirds
    setup_segs  = [s for s in transcript
                   if s["start"] >= start and s["start"] < start + third]
    payoff_segs = [s for s in transcript
                   if s["start"] >= end - third and s["start"] < end]

    setup_text  = " ".join(s["text"] for s in setup_segs).lower()
    payoff_text = " ".join(s["text"] for s in payoff_segs).lower()

    # Setup: orientation/question words, ideally starts a new sentence
    setup_hits  = sum(1 for w in SETUP_WORDS if w in setup_text.split())
    setup_score = min(setup_hits / 3.0, 1.0)
    if setup_text and setup_text[0].isupper():
        setup_score = min(setup_score + 0.25, 1.0)

    # Escalation: RMS peak in middle third vs full-window average
    si  = int(start + third)
    ei  = int(end - third)
    mid_rms  = float(np.mean(rms[si:ei])) if ei > si else 0.0
    full_rms = float(np.mean(rms[int(start):int(end)])) if int(end) > int(start) else 1e-6
    escalation_score = min(mid_rms / (full_rms + 1e-6), 1.0)

    # Payoff: exclamation marks, punchline words
    p_words     = payoff_text.split()
    payoff_hits = (
        payoff_text.count("!") * 0.25 +
        sum(0.15 for w in PAYOFF_WORDS if w in p_words)
    )
    payoff_score = min(payoff_hits, 1.0)

    return round(
        setup_score * 0.35 + escalation_score * 0.35 + payoff_score * 0.30,
        4
    )


def window_score(
    start: float,
    end: float,
    transcript: list[dict],
    rms: np.ndarray,
) -> tuple[float, str]:
    """Score a time window; returns (score 0–1, joined transcript text)."""
    words    = [s["text"] for s in transcript
                if s["end"] > start and s["start"] < end]
    joined     = " ".join(words).strip()
    word_count = len(joined.split())
    duration   = end - start

    if word_count < MIN_WORDS:
        return 0.0, joined

    dialogue_density = min(word_count / duration / 2.5, 1.0)
    si               = max(0, int(start))
    ei               = min(len(rms), int(np.ceil(end)))
    mean_energy      = float(np.mean(rms[si:ei])) if ei > si else 0.0
    arc              = narrative_arc_score(start, end, transcript, rms)

    score = 0.35 * dialogue_density + 0.30 * mean_energy + 0.35 * arc
    return round(score, 6), joined


def sliding_windows(
    transcript: list[dict],
    rms: np.ndarray,
    episode_duration: float,
) -> list[dict]:
    rms_max  = rms.max()
    rms_norm = rms / rms_max if rms_max > 0 else rms

    scored    = []
    t         = 30.0
    end_limit = episode_duration - 10.0

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

    scored.sort(key=lambda x: x["raw_score"], reverse=True)

    # Greedy dedup: keep highest-scoring, suppress >50% overlaps
    kept       = []
    suppressed = set()
    for i, w in enumerate(scored):
        if i in suppressed:
            continue
        kept.append(w)
        for j, other in enumerate(scored):
            if j <= i or j in suppressed:
                continue
            overlap  = max(0, min(w["end"], other["end"]) - max(w["start"], other["start"]))
            shorter  = min(w["duration"], other["duration"])
            if shorter > 0 and overlap / shorter > 0.5:
                suppressed.add(j)

    return kept[:MAX_CANDIDATES]


def _extract_code(video_path: str) -> str:
    stem = Path(video_path).stem
    m    = re.search(r"S(\d{2})E(\d{2})", stem, re.IGNORECASE)
    return f"S{m.group(1).upper()}E{m.group(2).upper()}" if m else stem


def run(video_path: str, output_dir: str) -> str:
    video_path   = str(Path(video_path).resolve())
    episode_stem = _extract_code(video_path)
    out_json     = os.path.join(output_dir, f"{episode_stem}_candidates_raw.json")

    print(f"\n[1_analyze] Episode : {episode_stem}  ({Path(video_path).name})")
    print(f"[1_analyze] Strategy: narrative-arc sliding window  ({WIN_SEC}s, step {STEP_SEC}s)")

    transcript_path = os.path.join(output_dir, f"{episode_stem}_transcript.json")
    words_path      = os.path.join(output_dir, f"{episode_stem}_words.json")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    try:
        print("  Extracting audio...")
        extract_audio(video_path, wav_path)

        if os.path.exists(transcript_path):
            print(f"  [cache] Reusing transcript: {transcript_path}")
            with open(transcript_path) as f:
                segments = json.load(f)
            episode_duration = segments[-1]["end"] if segments else 0
            print(f"  Episode duration: {episode_duration:.0f}s ({episode_duration/60:.1f} min)")

            if os.path.exists(words_path):
                print(f"  [cache] Reusing word timestamps: {words_path}")
            else:
                print("  Word timestamps missing — estimating from segments...")
                words = estimate_word_timestamps(segments)
                with open(words_path, "w") as f:
                    json.dump(words, f)
                print(f"  Saved {len(words)} word timestamps → {words_path}")
        else:
            segments, words = transcribe(wav_path)
            episode_duration = segments[-1]["end"] if segments else 0
            print(f"  {len(segments)} segments, {len(words)} word timestamps")
            print(f"  Episode duration: {episode_duration:.0f}s ({episode_duration/60:.1f} min)")

            with open(transcript_path, "w") as f:
                json.dump(segments, f)
            with open(words_path, "w") as f:
                json.dump(words, f)
            print(f"  Saved transcript + word timestamps")

        print("  Computing per-second RMS energy...")
        rms = compute_rms_per_second(wav_path, episode_duration)

        print("  Generating narrative-arc sliding windows...")
        candidates = sliding_windows(segments, rms, episode_duration)
        print(f"  Found {len(candidates)} candidate windows (top {MAX_CANDIDATES} after dedup)")

        payload = {
            "episode":    episode_stem,
            "video_path": video_path,
            "candidates": candidates,
        }
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"  Saved → {out_json}")
        return out_json

    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Analyze episode")
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output",  default="pipeline")
    args = parser.parse_args()
    run(args.episode, args.output)
