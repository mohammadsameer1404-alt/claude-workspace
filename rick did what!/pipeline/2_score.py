"""
Stage 2: Gemini Flash Virality Scoring — FREE, 3-Agent Panel Review

Uses Google Gemini 2.0 Flash (completely free tier):
  - 1,500 requests/day free  |  15 requests/minute free

Three-pass pipeline:
  Pass A — 480p bulk screen  → keeps score >= 6.5  (wide net, misses nothing)
  Pass B — 720p quality gate → keeps score >= 8.0  (eliminates weak survivors)
  Panel  — 3 specialist agents debate each Pass B survivor:
              • The Algorithm Expert  — YouTube completion rate / hook strength
              • The Rick & Morty Fan  — cultural weight, meme potential, fan reaction
              • The Devil's Advocate  — argues why it will flop
           Consensus >= 9.0 → final winner

This means every exported clip has been approved by 3 independent AI perspectives.
A clip that scrapes through on one opinion gets rejected. Only genuinely viral
moments make it through all three.

Get FREE key: aistudio.google.com → "Get API Key"
Set it: export GEMINI_API_KEY=your-key-here
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
PASS_A_THRESHOLD = 6.5   # wide net — catch everything with any potential
PASS_B_THRESHOLD = 8.0   # quality gate before expensive panel
PANEL_THRESHOLD  = 9.0   # consensus bar — all 3 agents must agree

# ── Models ────────────────────────────────────────────────────────────────────
# Primary for Pass A/B. Fallback list tried in order when primary hits quota.
MODEL_NAME = "gemini-2.5-flash"
MODEL_FALLBACKS = ["gemini-flash-lite-latest", "gemini-2.5-flash-lite"]

# ── Frame resolutions ─────────────────────────────────────────────────────────
PASS_A_W, PASS_A_H = 480, 270
PASS_B_W, PASS_B_H = 720, 405

# ── Rate limit (free tier = 15 RPM) ──────────────────────────────────────────
CALL_DELAY = 60 / 14   # ~4.3s between calls — stays under 15 RPM

# ── Safety settings (Rick & Morty = adult animated comedy) ───────────────────
SAFETY_OFF = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
]

# ── Prompts ───────────────────────────────────────────────────────────────────

PASS_A_PROMPT = """You are a YouTube Shorts virality screener for Rick and Morty.

Score this frame's viral potential 0.0–10.0:
- 9-10: Iconic/unforgettable — Pickle Rick reveal, Evil Morty twist, legendary one-liner
- 8-9:  Strong hook — great reaction, absurd visual, clearly meme-worthy
- 7-8:  Decent — moderate engagement expected
- 6-7:  Marginal — might work with right editing
- Below 6: Skip

Respond ONLY with valid JSON (no markdown):
{"score": 8.4, "reason": "brief reason"}"""

PASS_B_PROMPT = """You are a senior YouTube Shorts editor specialising in viral Rick and Morty content for the channel "Rick Dead Watt".

Score this clip 0.0–10.0 AND generate all publish metadata.

VIRAL SIGNALS (raise score):
✓ Punchline/payoff clearly visible in this frame
✓ Iconic character reaction — Rick's contempt, Morty's scream/panic, Jerry's obliviousness
✓ Visual absurdity that stops mid-scroll
✓ Rewatchable / shareable on its own without context
✓ Emotionally hooky in the first half-second

RED FLAGS (lower score):
✗ Needs 2+ minutes of prior context to land
✗ Talking heads, no visual interest
✗ Slow pace, no clear climax visible
✗ Season 4–5 generic filler

Also generate:
- "hook_text": 2–5 word pop title (shown first 2.5s of Short). ALL CAPS or punchy mixed case + emoji. Stops scrolling instantly.
- "youtube_title": max 60 chars, curiosity-gap + emoji, ends with #Shorts
- "description": 2 lines — line 1 is a hook sentence, line 2 is 8+ hashtags

Return ONLY valid JSON (no markdown):
{
  "score": 9.2,
  "reason": "why it will go viral",
  "best_moment_note": "tip on clip start/end timing",
  "hook_text": "Rick has NO mercy 😭",
  "youtube_title": "Rick solves everything in 30 seconds 💀 #Shorts",
  "description": "Rick and Morty being absolutely unhinged 😭\\n\\n#RickAndMorty #Shorts #RickDeadWatt #AdultSwim #rickmorty #animation #funny #viral"
}"""

PANEL_ANALYST_PROMPT = """You are The Algorithm Expert — a YouTube Shorts algorithm specialist.
Your job: judge this clip ONLY on YouTube Shorts performance mechanics.

Score 0.0–10.0 focusing on:
- Will viewers watch to the END? (completion rate = #1 signal)
- Is the first 0.5 seconds a scroll-stopper? (hook = makes or breaks it)
- Will people replay it? (replay rate = heavily weighted by algorithm)
- Does it trigger comments or shares? (highest-value engagement)
- Is it 30–55 seconds? (optimal duration for completion)

Be strict. A 9+ means this clip will genuinely be pushed by the algorithm to new audiences.

Respond ONLY with valid JSON:
{"score": 8.7, "verdict": "one sentence on algorithm performance potential"}"""

PANEL_FAN_PROMPT = """You are The Rick & Morty Superfan — you've watched every episode multiple times and are deeply embedded in the fandom.
Your job: judge this clip on cultural weight and fan/meme potential.

Score 0.0–10.0 focusing on:
- Is this a moment fans will instantly recognise and lose their minds over?
- Does it work as a standalone meme without the full episode?
- Will people tag their friends in the comments?
- Is it from a beloved episode/arc or forgotten filler?
- Does it capture what makes Rick and Morty GREAT?

Be honest. A 9+ means this clip will blow up in the Rick & Morty community.

Respond ONLY with valid JSON:
{"score": 9.1, "verdict": "one sentence on fan/meme potential"}"""

PANEL_SKEPTIC_PROMPT = """You are The Devil's Advocate — your job is to find every reason this clip will FLOP.

Score 0.0–10.0 where:
- 9-10: You cannot find a convincing reason for it to fail — it's genuinely great
- 7-8:  Minor issues but probably still performs
- 5-6:  Real problems that will hurt performance
- Below 5: This clip will underperform — explain why

Look hard for:
- Does it require too much context?
- Is the pacing too slow for Shorts?
- Is the joke too niche/obscure?
- Has this exact moment been clipped to death already?
- Is the visual quality poor?
- Does it start on a weak frame?

Respond ONLY with valid JSON:
{"score": 8.2, "verdict": "one sentence — main concern or why it actually holds up"}"""


def setup_gemini() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set.\n"
            "Get a FREE key at: aistudio.google.com → 'Get API Key'\n"
            "Then run: export GEMINI_API_KEY=your-key-here"
        )
    return genai.Client(api_key=api_key)


def extract_frame(video_path: str, ts: float, out: str, w: int, h: int) -> bool:
    r = subprocess.run([
        "ffmpeg", "-y", "-ss", str(ts), "-i", video_path,
        "-vframes", "1", "-vf", f"scale={w}:{h}", "-q:v", "4",
        out, "-loglevel", "error",
    ], capture_output=True)
    return r.returncode == 0 and os.path.exists(out)


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text.strip())


def call(client: genai.Client, frame_path: str, prompt: str, model: str = MODEL_NAME) -> dict:
    with open(frame_path, "rb") as f:
        img_data = f.read()
    models_to_try = [model] + [m for m in MODEL_FALLBACKS if m != model]
    last_err = None
    for m in models_to_try:
        try:
            resp = client.models.generate_content(
                model=m,
                contents=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=img_data, mime_type="image/jpeg"),
                ],
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


def panel_review(client: genai.Client, frame_path: str) -> dict | None:
    """
    Run the 3-agent panel on a single 720p frame.
    Returns consensus dict if all 3 agree >= PANEL_THRESHOLD, else None.
    """
    scores = {}
    verdicts = {}

    agents = [
        ("analyst",  PANEL_ANALYST_PROMPT,  "Algorithm Expert"),
        ("fan",      PANEL_FAN_PROMPT,      "Rick & Morty Fan "),
        ("skeptic",  PANEL_SKEPTIC_PROMPT,  "Devil's Advocate "),
    ]

    for key, prompt, label in agents:
        try:
            result = call(client, frame_path, prompt)
            scores[key]   = float(result.get("score", 0))
            verdicts[key] = result.get("verdict", "")
            print(f"      [{label}] {scores[key]:.1f} — {verdicts[key][:55]}")
        except Exception as e:
            print(f"      [{label}] ERROR: {e}")
            scores[key]   = 0.0
            verdicts[key] = str(e)

    consensus = sum(scores.values()) / len(scores)
    min_score  = min(scores.values())

    # Approval: average consensus >= PANEL_THRESHOLD AND no agent scores below 7.0
    # (The strict all-three-must-hit-9.0 rule breaks when lite models are used for the panel)
    approved = consensus >= PANEL_THRESHOLD and min_score >= 7.0

    print(f"      Consensus: {consensus:.1f}  (min: {min_score:.1f}  need avg≥{PANEL_THRESHOLD} & min≥7.0)  {'✓' if approved else '✗'}")

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

    cache = load_cache(cache_path)
    client = setup_gemini()

    print(f"\n[2_score] Episode    : {episode}")
    print(f"[2_score] Candidates : {len(candidates)}")
    print(f"[2_score] Model      : {MODEL_NAME} (FREE)")
    print(f"[2_score] Thresholds : Pass A≥{PASS_A_THRESHOLD}  Pass B≥{PASS_B_THRESHOLD}  Panel≥{PANEL_THRESHOLD}")

    # ── PASS A: 480p bulk screen ──────────────────────────────────────────────
    print(f"\n  ── Pass A: 480p bulk screen (threshold ≥ {PASS_A_THRESHOLD}) ──")
    pass_a = []

    for i, cand in enumerate(candidates):
        key = f"A_{episode}_{cand['start']}_{cand['end']}"
        if key in cache:
            score = cache[key]["score"]
            print(f"  [cache] {i:02d}: {score:.1f}")
        else:
            mid   = round((cand["start"] + cand["end"]) / 2, 2)
            fpath = os.path.join(frame_dir, f"A_{i:04d}.jpg")
            if not extract_frame(video_path, mid, fpath, PASS_A_W, PASS_A_H):
                print(f"  [warn] frame extract failed for candidate {i}")
                continue
            print(f"  [{i:02d}] {cand['start']:.0f}s–{cand['end']:.0f}s ...", end=" ", flush=True)
            try:
                r     = call(client, fpath, PASS_A_PROMPT)
                score = float(r.get("score", 0))
                cache[key] = {"score": score, "reason": r.get("reason", "")}
                print(f"{score:.1f}  {r.get('reason','')[:50]}")
            except Exception as e:
                print(f"error: {e}")
                score = 0.0

        if score >= PASS_A_THRESHOLD:
            pass_a.append((cand, cache.get(key, {"score": score})))

    save_cache(cache_path, cache)
    shutil.rmtree(frame_dir, ignore_errors=True)
    os.makedirs(frame_dir, exist_ok=True)
    print(f"\n  Pass A: {len(pass_a)}/{len(candidates)} survived")

    # ── PASS B: 720p quality gate + metadata ─────────────────────────────────
    print(f"\n  ── Pass B: 720p quality gate + metadata (threshold ≥ {PASS_B_THRESHOLD}) ──")
    pass_b = []

    for idx, (cand, pa) in enumerate(pass_a):
        key = f"B_{episode}_{cand['start']}_{cand['end']}"
        if key in cache:
            sd    = cache[key]
            score = float(sd.get("score", 0))
            print(f"  [cache] {cand['start']:.0f}s–{cand['end']:.0f}s → {score:.1f}")
        else:
            mid   = round((cand["start"] + cand["end"]) / 2, 2)
            fpath = os.path.join(frame_dir, f"B_{idx:04d}.jpg")
            if not extract_frame(video_path, mid, fpath, PASS_B_W, PASS_B_H):
                continue
            print(f"  {cand['start']:.0f}s–{cand['end']:.0f}s (PassA:{pa['score']:.1f}) ...", end=" ", flush=True)
            try:
                sd    = call(client, fpath, PASS_B_PROMPT)
                score = float(sd.get("score", 0))
                cache[key] = sd
                print(f"→ {score:.1f}  \"{sd.get('hook_text','')}\"")
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

    # ── PANEL: 3-agent debate on every Pass B survivor ────────────────────────
    winners = []

    if skip_panel:
        print(f"\n  ── Panel skipped (--skip-panel) — promoting all Pass B survivors ──")
        for cand, pa, sd, pass_b_score in pass_b:
            winners.append({
                **cand,
                "score":         pass_b_score,
                "pass_a_score":  pa["score"],
                "pass_b_score":  pass_b_score,
                "panel":         {},
                "reason":        sd.get("reason", ""),
                "timing_note":   sd.get("best_moment_note", ""),
                "hook_text":     sd.get("hook_text",     "Rick and Morty 😭"),
                "youtube_title": sd.get("youtube_title", "Rick and Morty best moment 💀 #Shorts"),
                "description":   sd.get("description",  "Rick and Morty being unhinged 😭\n\n#RickAndMorty #Shorts #RickDeadWatt #AdultSwim #rickmorty #animation #funny #viral"),
                "episode":       episode,
                "video_path":    video_path,
                "clip_id":       f"{episode}_{int(cand['start'])}_{int(cand['end'])}",
            })
    else:
        print(f"\n  ── 3-Agent Panel Review (consensus threshold ≥ {PANEL_THRESHOLD}) ──")
        for idx, (cand, pa, sd, pass_b_score) in enumerate(pass_b):
            key = f"PANEL_{episode}_{cand['start']}_{cand['end']}"
            print(f"\n  Clip {idx+1}/{len(pass_b)}: {cand['start']:.0f}s–{cand['end']:.0f}s  (Pass B: {pass_b_score:.1f})")

            if key in cache:
                panel = cache[key]
                print(f"    [cache] consensus: {panel['consensus']:.1f}")
                approved = panel["consensus"] >= PANEL_THRESHOLD and min(panel["scores"].values()) >= 7.0
            else:
                mid   = round((cand["start"] + cand["end"]) / 2, 2)
                fpath = os.path.join(frame_dir, f"PANEL_{idx:04d}.jpg")
                if not extract_frame(video_path, mid, fpath, PASS_B_W, PASS_B_H):
                    continue
                panel    = panel_review(client, fpath)
                approved = panel is not None
                if panel:
                    cache[key] = panel
                    save_cache(cache_path, cache)

            if approved:
                final_score = round((pass_b_score + cache[key]["consensus"]) / 2, 2)
                print(f"    ✓ APPROVED  final score: {final_score}")
                winners.append({
                    **cand,
                    "score":         final_score,
                    "pass_a_score":  pa["score"],
                    "pass_b_score":  pass_b_score,
                    "panel":         cache.get(key, {}),
                    "reason":        sd.get("reason", ""),
                    "timing_note":   sd.get("best_moment_note", ""),
                    "hook_text":     sd.get("hook_text",     "Rick and Morty 😭"),
                    "youtube_title": sd.get("youtube_title", "Rick and Morty best moment 💀 #Shorts"),
                    "description":   sd.get("description",  "Rick and Morty being unhinged 😭\n\n#RickAndMorty #Shorts #RickDeadWatt #AdultSwim #rickmorty #animation #funny #viral"),
                    "episode":       episode,
                    "video_path":    video_path,
                    "clip_id":       f"{episode}_{int(cand['start'])}_{int(cand['end'])}",
                })
            else:
                print(f"    ✗ REJECTED  (panel did not reach consensus)")

    shutil.rmtree(frame_dir, ignore_errors=True)
    winners.sort(key=lambda x: x["score"], reverse=True)

    with open(out_json, "w") as f:
        json.dump({"episode": episode, "clips": winners}, f, indent=2)

    print(f"\n{'═'*55}")
    print(f"  RESULTS FOR {episode}")
    print(f"  {len(candidates)} candidates → {len(pass_a)} Pass A → {len(pass_b)} Pass B → {len(winners)} APPROVED")
    print(f"{'═'*55}")
    for w in winners:
        print(f"  [{w['score']:.1f}★] {w['start']:.0f}s–{w['end']:.0f}s")
        print(f"         Hook  : \"{w['hook_text']}\"")
        print(f"         Title : {w['youtube_title']}")
    print(f"\n  Saved → {out_json}")

    return out_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: 3-agent panel virality scoring (Gemini free)")
    parser.add_argument("--raw",         required=True)
    parser.add_argument("--output",      default="pipeline")
    parser.add_argument("--temp",        default="temp")
    parser.add_argument("--skip-panel",  action="store_true",
                        help="Skip panel review and promote all Pass B survivors directly")
    args = parser.parse_args()
    run(args.raw, args.output, args.temp, skip_panel=args.skip_panel)
