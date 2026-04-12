"""
Stage 2: Gemini Flash 3-Agent Panel Scoring — FREE, Story-Arc Focused

Changes from v1:
  - 3 frames per clip (HOOK at +2s, MIDDLE at midpoint, PAYOFF at -3s)
  - All prompts rewritten to require complete story arc (start→mid→end)
  - Pass B adds arc_score field; arc<8.0 caps overall score at 7.5
  - Thresholds raised: Pass A≥7.0, Pass B≥8.5, Panel≥8.5
  - Hard cap: top 3 clips per episode only
  - Shorter prompts — fewer tokens, same accuracy
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from google import genai
from google.genai import types

# ── Thresholds ────────────────────────────────────────────────────────────────
PASS_A_THRESHOLD = 7.0   # wider screen but arc-aware
PASS_B_THRESHOLD = 8.5   # user minimum: 8.5
PANEL_THRESHOLD  = 8.5   # consensus bar

MAX_CLIPS_PER_EPISODE = 3  # hard cap

# ── Models ────────────────────────────────────────────────────────────────────
MODEL_NAME     = "gemini-2.5-flash"
MODEL_FALLBACKS = ["gemini-flash-lite-latest", "gemini-2.5-flash-lite"]

# ── Frame resolutions ─────────────────────────────────────────────────────────
PASS_A_W, PASS_A_H = 480, 270
PASS_B_W, PASS_B_H = 720, 405

# ── Rate limit (free tier = 15 RPM) ──────────────────────────────────────────
CALL_DELAY = 60 / 14   # ~4.3s — stays under 15 RPM

# ── Safety settings ───────────────────────────────────────────────────────────
SAFETY_OFF = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
]

# ── Prompts ───────────────────────────────────────────────────────────────────

def _transcript_snippet(text: str, max_chars: int = 150) -> str:
    return text[:max_chars].rsplit(" ", 1)[0] + "…" if len(text) > max_chars else text


def pass_a_prompt(transcript: str) -> str:
    t = _transcript_snippet(transcript)
    return f"""Rick & Morty Short. Image 1=HOOK(2s in), Image 2=MIDDLE, Image 3=PAYOFF(final 3s).
Transcript: "{t}"

Score 0-10. REQUIRE all 3:
• Hook grabs in 0.5s (scroll-stopper)
• Middle escalates or builds tension/comedy
• Payoff lands — clear punchline, reaction, or resolution
FAIL if: starts mid-sentence, ends without resolution, needs prior context to land.
JSON only: {{"score":8.4,"reason":"brief"}}"""


def pass_b_prompt(transcript: str) -> str:
    t = _transcript_snippet(transcript)
    return f"""Senior R&M Shorts editor. Image 1=HOOK, Image 2=MIDDLE, Image 3=PAYOFF.
Transcript: "{t}"

Score 0-10 overall. Also score separately: hook(0-10), arc(0-10), payoff(0-10).
Rule: arc<8.0 → cap overall at 7.5 regardless.
Generate: hook_text(2-5 words ALL CAPS+emoji, stops scrolling), youtube_title(≤60 chars #Shorts), description(1 hook sentence + 8 hashtags).
JSON only: {{"score":9.2,"hook":9.1,"arc":8.8,"payoff":9.0,"reason":"why viral","hook_text":"TEXT 😭","youtube_title":"Title #Shorts","description":"Hook.\\n\\n#RickAndMorty #Shorts #RickDeadWatt #AdultSwim #rickmorty #animation #funny #viral"}}"""


def panel_analyst_prompt(transcript: str) -> str:
    t = _transcript_snippet(transcript)
    return f"""YouTube algo expert. R&M Short. Image 1=HOOK, Image 2=MID, Image 3=END.
Transcript: "{t}"
Judge: completion rate, 0.5s scroll-stop, replay value, satisfying ending.
Strict: 9+=genuinely viral. JSON: {{"score":8.7,"verdict":"one sentence"}}"""


def panel_fan_prompt(transcript: str) -> str:
    t = _transcript_snippet(transcript)
    return f"""R&M superfan judge. Image 1=HOOK, Image 2=MID, Image 3=END.
Transcript: "{t}"
Judge: fan recognition, standalone meme, iconic moment, no context needed.
JSON: {{"score":9.1,"verdict":"one sentence"}}"""


def panel_skeptic_prompt(transcript: str) -> str:
    t = _transcript_snippet(transcript)
    return f"""Devil's advocate. Find every flaw. R&M Short. Image 1=HOOK, Image 2=MID, Image 3=END.
Transcript: "{t}"
Attack: weak ending, needs context, niche, slow, already over-clipped.
9+=can't find real flaws. JSON: {{"score":8.2,"verdict":"main concern"}}"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def setup_gemini() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set.\n"
            "Get FREE key: aistudio.google.com → 'Get API Key'\n"
            "Then: export GEMINI_API_KEY=your-key"
        )
    return genai.Client(api_key=api_key)


def extract_frame(video_path: str, ts: float, out: str, w: int, h: int) -> bool:
    r = subprocess.run([
        "ffmpeg", "-y", "-ss", str(ts), "-i", video_path,
        "-vframes", "1", "-vf", f"scale={w}:{h}", "-q:v", "4",
        out, "-loglevel", "error",
    ], capture_output=True)
    return r.returncode == 0 and os.path.exists(out)


def extract_three_frames(
    video_path: str,
    start: float,
    end: float,
    frame_dir: str,
    prefix: str,
    w: int,
    h: int,
) -> tuple[str, str, str] | None:
    """Extract hook (+2s), mid, payoff (-3s) frames. Returns (hook, mid, payoff) paths or None."""
    hook_ts    = start + 2.0
    mid_ts     = (start + end) / 2
    payoff_ts  = max(start + 2, end - 3.0)

    hook_path    = os.path.join(frame_dir, f"{prefix}_hook.jpg")
    mid_path     = os.path.join(frame_dir, f"{prefix}_mid.jpg")
    payoff_path  = os.path.join(frame_dir, f"{prefix}_payoff.jpg")

    if not extract_frame(video_path, hook_ts,   hook_path,   w, h): return None
    if not extract_frame(video_path, mid_ts,    mid_path,    w, h): return None
    if not extract_frame(video_path, payoff_ts, payoff_path, w, h): return None

    return hook_path, mid_path, payoff_path


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text  = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text.strip())


def call(
    client: genai.Client,
    frame_paths: list[str],
    prompt: str,
    model: str = MODEL_NAME,
) -> dict:
    """Send prompt + 1-3 frames to Gemini. Returns parsed JSON dict."""
    models_to_try = [model] + [m for m in MODEL_FALLBACKS if m != model]
    parts = [types.Part.from_text(text=prompt)]
    for fp in frame_paths:
        with open(fp, "rb") as f:
            parts.append(types.Part.from_bytes(data=f.read(), mime_type="image/jpeg"))

    last_err = None
    for m in models_to_try:
        try:
            resp = client.models.generate_content(
                model=m,
                contents=parts,
                config=types.GenerateContentConfig(safety_settings=SAFETY_OFF),
            )
            time.sleep(CALL_DELAY)
            return parse_json(resp.text)
        except Exception as e:
            if "429" in str(e):
                print(f" [quota on {m}, trying next]", end=" ", flush=True)
                last_err = e
                continue
            raise
    raise last_err


def panel_review(client: genai.Client, frames: tuple[str, str, str], transcript: str) -> dict | None:
    """3-agent panel. Returns consensus dict if approved, else None."""
    scores   = {}
    verdicts = {}

    agents = [
        ("analyst",  panel_analyst_prompt(transcript),  "Algorithm Expert"),
        ("fan",      panel_fan_prompt(transcript),      "R&M Superfan    "),
        ("skeptic",  panel_skeptic_prompt(transcript),  "Devil's Advocate"),
    ]

    for key, prompt, label in agents:
        try:
            result       = call(client, list(frames), prompt)
            scores[key]  = float(result.get("score", 0))
            verdicts[key] = result.get("verdict", "")
            print(f"      [{label}] {scores[key]:.1f} — {verdicts[key][:55]}")
        except Exception as e:
            print(f"      [{label}] ERROR: {e}")
            scores[key]   = 0.0
            verdicts[key] = str(e)

    consensus = sum(scores.values()) / len(scores)
    min_score  = min(scores.values())
    approved   = consensus >= PANEL_THRESHOLD and min_score >= 7.0

    print(f"      Consensus: {consensus:.1f}  min:{min_score:.1f}  "
          f"(need avg≥{PANEL_THRESHOLD} & min≥7.0)  {'✓' if approved else '✗'}")

    if approved:
        return {"consensus": round(consensus, 2), "scores": scores, "verdicts": verdicts}
    return None


def load_cache(path: str) -> dict:
    return json.loads(Path(path).read_text()) if os.path.exists(path) else {}


def save_cache(path: str, cache: dict) -> None:
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def run(raw_json_path: str, output_dir: str, temp_dir: str, skip_panel: bool = False) -> str:
    with open(raw_json_path) as f:
        data = json.load(f)

    episode    = data["episode"]
    video_path = data["video_path"]
    candidates = data["candidates"]

    cache_path = os.path.join(output_dir, "scorer_cache.json")
    out_json   = os.path.join(output_dir, f"{episode}_candidates.json")
    frame_dir  = os.path.join(temp_dir, f"{episode}_frames")
    os.makedirs(frame_dir, exist_ok=True)

    cache  = load_cache(cache_path)
    client = setup_gemini()

    print(f"\n[2_score] Episode    : {episode}")
    print(f"[2_score] Candidates : {len(candidates)}")
    print(f"[2_score] Model      : {MODEL_NAME} (FREE)")
    print(f"[2_score] Thresholds : Pass A≥{PASS_A_THRESHOLD}  Pass B≥{PASS_B_THRESHOLD}  Panel≥{PANEL_THRESHOLD}")
    print(f"[2_score] Max clips  : {MAX_CLIPS_PER_EPISODE} per episode")

    # ── PASS A ────────────────────────────────────────────────────────────────
    print(f"\n  ── Pass A: {PASS_A_W}p bulk screen (threshold ≥ {PASS_A_THRESHOLD}) ──")
    pass_a = []

    for i, cand in enumerate(candidates):
        key = f"A_{episode}_{cand['start']}_{cand['end']}"
        if key in cache:
            score = cache[key]["score"]
            print(f"  [cache] {i:02d}: {score:.1f}")
        else:
            prefix = f"A_{i:04d}"
            frames = extract_three_frames(
                video_path, cand["start"], cand["end"],
                frame_dir, prefix, PASS_A_W, PASS_A_H
            )
            if not frames:
                print(f"  [warn] frame extract failed for candidate {i}")
                continue
            print(f"  [{i:02d}] {cand['start']:.0f}s–{cand['end']:.0f}s ...", end=" ", flush=True)
            try:
                r     = call(client, list(frames), pass_a_prompt(cand["transcript"]))
                score = float(r.get("score", 0))
                cache[key] = {"score": score, "reason": r.get("reason", "")}
                print(f"{score:.1f}  {r.get('reason','')[:55]}")
            except Exception as e:
                print(f"error: {e}")
                score = 0.0

        if score >= PASS_A_THRESHOLD:
            pass_a.append((cand, cache.get(key, {"score": score})))

    save_cache(cache_path, cache)
    shutil.rmtree(frame_dir, ignore_errors=True)
    os.makedirs(frame_dir, exist_ok=True)
    print(f"\n  Pass A: {len(pass_a)}/{len(candidates)} survived")

    # ── PASS B ────────────────────────────────────────────────────────────────
    print(f"\n  ── Pass B: {PASS_B_W}p quality gate + metadata (threshold ≥ {PASS_B_THRESHOLD}) ──")
    pass_b = []

    for idx, (cand, pa) in enumerate(pass_a):
        key = f"B_{episode}_{cand['start']}_{cand['end']}"
        if key in cache:
            sd    = cache[key]
            score = float(sd.get("score", 0))
            # Apply arc cap from cache if present
            arc_score = float(sd.get("arc", 10))
            if arc_score < 8.0:
                score = min(score, 7.5)
            print(f"  [cache] {cand['start']:.0f}s–{cand['end']:.0f}s → {score:.1f}")
        else:
            prefix = f"B_{idx:04d}"
            frames = extract_three_frames(
                video_path, cand["start"], cand["end"],
                frame_dir, prefix, PASS_B_W, PASS_B_H
            )
            if not frames:
                continue
            print(f"  {cand['start']:.0f}s–{cand['end']:.0f}s (A:{pa['score']:.1f}) ...",
                  end=" ", flush=True)
            try:
                sd        = call(client, list(frames), pass_b_prompt(cand["transcript"]))
                score     = float(sd.get("score", 0))
                arc_score = float(sd.get("arc", 10))
                if arc_score < 8.0:
                    score = min(score, 7.5)
                    sd["score"] = score
                cache[key] = sd
                print(f"→ {score:.1f}  arc:{arc_score:.1f}  \"{sd.get('hook_text','')}\"")
            except Exception as e:
                print(f"error: {e}")
                score = pa["score"]
                sd    = pa
        save_cache(cache_path, cache)

        if score >= PASS_B_THRESHOLD:
            pass_b.append((cand, pa, sd, score))

    shutil.rmtree(frame_dir, ignore_errors=True)
    os.makedirs(frame_dir, exist_ok=True)
    print(f"\n  Pass B: {len(pass_b)}/{len(pass_a)} survived")

    # ── PANEL ─────────────────────────────────────────────────────────────────
    winners = []

    if skip_panel:
        print(f"\n  ── Panel skipped — promoting all Pass B survivors ──")
        for cand, pa, sd, pass_b_score in pass_b:
            winners.append(_build_winner(cand, pa, sd, pass_b_score, {}, episode, video_path))
    else:
        print(f"\n  ── 3-Agent Panel (consensus threshold ≥ {PANEL_THRESHOLD}) ──")
        for idx, (cand, pa, sd, pass_b_score) in enumerate(pass_b):
            key = f"PANEL_{episode}_{cand['start']}_{cand['end']}"
            print(f"\n  Clip {idx+1}/{len(pass_b)}: {cand['start']:.0f}s–{cand['end']:.0f}s  (Pass B: {pass_b_score:.1f})")

            if key in cache:
                panel    = cache[key]
                approved = panel["consensus"] >= PANEL_THRESHOLD and min(panel["scores"].values()) >= 7.0
                print(f"    [cache] consensus: {panel['consensus']:.1f}  {'✓' if approved else '✗'}")
            else:
                prefix = f"PANEL_{idx:04d}"
                frames = extract_three_frames(
                    video_path, cand["start"], cand["end"],
                    frame_dir, prefix, PASS_B_W, PASS_B_H
                )
                if not frames:
                    continue
                panel    = panel_review(client, frames, cand["transcript"])
                approved = panel is not None
                if panel:
                    cache[key] = panel
                    save_cache(cache_path, cache)

            if approved:
                final_score = round((pass_b_score + cache[key]["consensus"]) / 2, 2)
                print(f"    ✓ APPROVED  final score: {final_score}")
                winners.append(
                    _build_winner(cand, pa, sd, pass_b_score, cache.get(key, {}),
                                  episode, video_path, final_score)
                )
            else:
                print(f"    ✗ REJECTED")

    shutil.rmtree(frame_dir, ignore_errors=True)

    # ── Hard cap: top 3 per episode ───────────────────────────────────────────
    winners.sort(key=lambda x: x["score"], reverse=True)
    if len(winners) > MAX_CLIPS_PER_EPISODE:
        print(f"\n  Hard cap: keeping top {MAX_CLIPS_PER_EPISODE} of {len(winners)} approved clips")
        winners = winners[:MAX_CLIPS_PER_EPISODE]

    with open(out_json, "w") as f:
        json.dump({"episode": episode, "clips": winners}, f, indent=2)

    print(f"\n{'═'*55}")
    print(f"  RESULTS FOR {episode}")
    print(f"  {len(candidates)} candidates → {len(pass_a)} Pass A → {len(pass_b)} Pass B → {len(winners)} FINAL")
    print(f"{'═'*55}")
    for w in winners:
        print(f"  [{w['score']:.1f}★] {w['start']:.0f}s–{w['end']:.0f}s")
        print(f"         Hook  : \"{w['hook_text']}\"")
        print(f"         Title : {w['youtube_title']}")
    print(f"\n  Saved → {out_json}")
    return out_json


def _build_winner(
    cand: dict,
    pa: dict,
    sd: dict,
    pass_b_score: float,
    panel: dict,
    episode: str,
    video_path: str,
    final_score: float | None = None,
) -> dict:
    if final_score is None:
        final_score = pass_b_score
    return {
        **cand,
        "score":         final_score,
        "pass_a_score":  pa.get("score", 0),
        "pass_b_score":  pass_b_score,
        "panel":         panel,
        "reason":        sd.get("reason", ""),
        "timing_note":   sd.get("best_moment_note", ""),
        "hook_text":     sd.get("hook_text",     "Rick and Morty 😭"),
        "youtube_title": sd.get("youtube_title", "Rick and Morty moment 💀 #Shorts"),
        "description":   sd.get("description",
                                "Rick and Morty being unhinged 😭\n\n"
                                "#RickAndMorty #Shorts #RickDeadWatt #AdultSwim "
                                "#rickmorty #animation #funny #viral"),
        "episode":    episode,
        "video_path": video_path,
        "clip_id":    f"{episode}_{int(cand['start'])}_{int(cand['end'])}",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: 3-agent panel scoring (Gemini FREE)")
    parser.add_argument("--raw",        required=True)
    parser.add_argument("--output",     default="pipeline")
    parser.add_argument("--temp",       default="temp")
    parser.add_argument("--skip-panel", action="store_true")
    args = parser.parse_args()
    run(args.raw, args.output, args.temp, skip_panel=args.skip_panel)
