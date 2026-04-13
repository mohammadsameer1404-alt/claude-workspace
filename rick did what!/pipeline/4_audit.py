"""
Stage 4: 3-AI Independent Virality Audit — All FREE

Three genuinely different AI companies evaluate each exported clip:
  1. Google Gemini 2.5 Flash  — vision + text (GEMINI_API_KEY, already set up)
  2. Groq / Llama 3.3 70B     — text only     (GROQ_API_KEY,  free at console.groq.com)
  3. Mistral Small             — text only     (MISTRAL_API_KEY, free at console.mistral.ai)

Token strategy: minimal prompts (<70 words), capped transcript (120 chars), ~30 output
tokens per call. Total per clip: ~90 output tokens across all 3 auditors.

Missing API key → that auditor is skipped gracefully.
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

# ── Frame extraction ──────────────────────────────────────────────────────────
FRAME_W, FRAME_H = 480, 270   # small frame — enough for Gemini vision audit


def extract_payoff_frame(video_path: str, out_path: str) -> bool:
    """Extract a single frame 3s from the end of the clip (the payoff moment)."""
    # Get clip duration first
    dur_out = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "csv=p=0", video_path,
    ], capture_output=True, text=True)
    try:
        duration = float(dur_out.stdout.strip())
    except ValueError:
        duration = 40.0  # fallback

    ts = max(0, duration - 3.0)
    r  = subprocess.run([
        "ffmpeg", "-y", "-ss", str(ts), "-i", video_path,
        "-vframes", "1", "-vf", f"scale={FRAME_W}:{FRAME_H}", "-q:v", "4",
        out_path, "-loglevel", "error",
    ], capture_output=True)
    return r.returncode == 0 and os.path.exists(out_path)


def _snippet(text: str, max_chars: int = 120) -> str:
    return text[:max_chars].rsplit(" ", 1)[0] + "…" if len(text) > max_chars else text


# ── Prompts ───────────────────────────────────────────────────────────────────

def _audit_prompt(duration: float, transcript: str) -> str:
    t = _snippet(transcript)
    return (
        f"R&M Short. {duration:.0f}s. Dialogue: \"{t}\"\n"
        "Score viral potential 0-10. 9+=viral, 8+=strong, <7=skip.\n"
        "Check: hook strength, complete arc (setup→mid→payoff), satisfying end, no context needed.\n"
        'JSON only: {"score":8.7,"viral":true,"verdict":"10 words max","flaw":"or null"}'
    )


# ── Auditor 1: Gemini (vision + text) ────────────────────────────────────────

def audit_gemini(
    frame_path: str | None,
    duration: float,
    transcript: str,
) -> dict | None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        prompt = _audit_prompt(duration, transcript)
        contents = [types.Part.from_text(text=prompt)]
        if frame_path and os.path.exists(frame_path):
            with open(frame_path, "rb") as f:
                contents.append(types.Part.from_bytes(data=f.read(), mime_type="image/jpeg"))

        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                safety_settings=[
                    types.SafetySetting(category=c, threshold="BLOCK_NONE")
                    for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                               "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
                ]
            ),
        )
        return _parse(resp.text)
    except Exception as e:
        return {"error": str(e)}


# ── Auditor 2: Groq / Llama (text only) ──────────────────────────────────────

def audit_groq(duration: float, transcript: str) -> dict | None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        prompt = _audit_prompt(duration, transcript)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.2,
        )
        return _parse(resp.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}


# ── Auditor 3: Mistral (text only) ───────────────────────────────────────────

def audit_mistral(duration: float, transcript: str) -> dict | None:
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        return None
    try:
        from mistralai.client import Mistral
        client = Mistral(api_key=api_key)
        prompt = _audit_prompt(duration, transcript)
        resp = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.2,
        )
        return _parse(resp.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}


# ── JSON parser ───────────────────────────────────────────────────────────────

def _parse(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text  = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    # Find first { ... } block
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]
    return json.loads(text)


# ── Report printer ────────────────────────────────────────────────────────────

def _print_report(clip_id: str, results: dict) -> dict:
    auditors = [
        ("Gemini 2.5 Flash", results.get("gemini")),
        ("Groq Llama 3.3  ", results.get("groq")),
        ("Mistral Small   ", results.get("mistral")),
    ]

    ran     = [(name, r) for name, r in auditors if r and "score" in r]
    skipped = [(name, r) for name, r in auditors if r is None]
    errored = [(name, r) for name, r in auditors if r and "error" in r]

    scores  = [float(r["score"]) for _, r in ran]
    consensus = round(sum(scores) / len(scores), 2) if scores else 0.0
    all_viral = all(r.get("viral", False) for _, r in ran if "viral" in r)

    W = 60
    print(f"\n╔{'═'*W}╗")
    print(f"║  VIRALITY AUDIT — {clip_id:<{W-20}}║")
    print(f"╠{'═'*W}╣")
    for name, r in ran:
        score   = float(r["score"])
        viral   = "YES" if r.get("viral") else "NO "
        verdict = r.get("verdict", "")[:28]
        mark    = "✓" if score >= 8.5 else "✗"
        print(f"║  {name} │ {score:.1f} {mark} │ viral:{viral} │ {verdict:<28}║")
    for name, _ in skipped:
        print(f"║  {name} │ SKIPPED (no API key){' '*(W-31)}║")
    for name, r in errored:
        print(f"║  {name} │ ERROR: {str(r.get('error',''))[:40]}{' '*(W-49)}║")
    print(f"╠{'═'*W}╣")
    if scores:
        status = "✅ APPROVED" if consensus >= 8.5 and all_viral else "⚠️  REVIEW"
        print(f"║  CONSENSUS: {consensus:.2f} — {status:<{W-18}}║")
    else:
        print(f"║  CONSENSUS: N/A — no auditors ran{' '*(W-35)}║")
    print(f"╚{'═'*W}╝")

    # Flag weaknesses
    for name, r in ran:
        flaw = r.get("flaw")
        if flaw and flaw != "null" and flaw is not None:
            print(f"  ⚠️  {name.strip()}: \"{flaw}\"")

    return {
        "clip_id":   clip_id,
        "consensus": consensus,
        "approved":  consensus >= 8.5 and bool(scores),
        "auditors":  {k: v for k, v in results.items() if v},
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def audit_clip(clip_path: str, candidates_json: str | None = None) -> dict:
    """Run 3-AI audit on a single exported clip MP4."""
    clip_id = Path(clip_path).stem

    # Load transcript from candidates JSON if available
    transcript = ""
    duration   = 40.0
    if candidates_json and os.path.exists(candidates_json):
        with open(candidates_json) as f:
            data = json.load(f)
        for clip in data.get("clips", []):
            if clip.get("clip_id") == clip_id:
                transcript = clip.get("transcript", "")
                duration   = float(clip.get("end", 0)) - float(clip.get("start", 0))
                break

    # Get actual duration from file if not from JSON
    if duration <= 0:
        dur_out = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", clip_path,
        ], capture_output=True, text=True)
        try:
            duration = float(dur_out.stdout.strip())
        except ValueError:
            duration = 40.0

    print(f"\n[4_audit] Clip: {clip_id}  ({duration:.0f}s)")
    print(f"[4_audit] Running 3 independent AI auditors...")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        frame_path = f.name

    try:
        frame_ok = extract_payoff_frame(clip_path, frame_path)

        results = {}

        print("  [1/3] Gemini 2.5 Flash (vision + text)...", end=" ", flush=True)
        results["gemini"] = audit_gemini(frame_path if frame_ok else None, duration, transcript)
        _print_auditor_line(results["gemini"])

        print("  [2/3] Groq / Llama 3.3 70B (text)...", end=" ", flush=True)
        results["groq"] = audit_groq(duration, transcript)
        _print_auditor_line(results["groq"])

        print("  [3/3] Mistral Small (text)...", end=" ", flush=True)
        results["mistral"] = audit_mistral(duration, transcript)
        _print_auditor_line(results["mistral"])

    finally:
        if os.path.exists(frame_path):
            os.unlink(frame_path)

    return _print_report(clip_id, results)


def _print_auditor_line(result: dict | None) -> None:
    if result is None:
        print("SKIPPED (no key)")
    elif "error" in result:
        print(f"ERROR — {result['error'][:60]}")
    else:
        score = result.get("score", "?")
        print(f"{score}/10  viral:{result.get('viral', '?')}")


def run(clips_dir: str, pipeline_dir: str) -> list[dict]:
    clip_files = sorted(Path(clips_dir).glob("*.mp4"))
    if not clip_files:
        print("[4_audit] No clips found in", clips_dir)
        return []

    print(f"\n[4_audit] Auditing {len(clip_files)} clip(s)...")
    print("[4_audit] APIs: Gemini(free) + Groq(free) + Mistral(free)")

    # Find candidates JSON — supports creative slugs via manifest.json
    manifest_path = os.path.join(clips_dir, "manifest.json")
    manifest: dict = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as _mf:
            manifest = json.load(_mf)

    def find_candidates(clip_id: str) -> str | None:
        # Creative slug → look up episode in manifest
        if clip_id in manifest:
            episode = manifest[clip_id]["episode"]
            path    = os.path.join(pipeline_dir, f"{episode}_candidates.json")
            return path if os.path.exists(path) else None
        # Fallback: technical ID format e.g. S01E05_340_385
        episode = "_".join(clip_id.split("_")[:2])
        path    = os.path.join(pipeline_dir, f"{episode}_candidates.json")
        return path if os.path.exists(path) else None

    reports    = []
    audit_log  = os.path.join(pipeline_dir, "audit_log.json")

    for clip_path in clip_files:
        clip_id   = clip_path.stem
        cand_json = find_candidates(clip_id)
        report    = audit_clip(str(clip_path), cand_json)
        reports.append(report)

    with open(audit_log, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"\n[4_audit] Audit log → {audit_log}")

    approved = sum(1 for r in reports if r.get("approved"))
    print(f"[4_audit] {approved}/{len(reports)} clips approved by all auditors ✓")
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: 3-AI virality audit (all free)")
    parser.add_argument("--clips",    required=True, help="Directory of exported .mp4 clips")
    parser.add_argument("--pipeline", required=True, help="Pipeline directory with candidates JSON")
    args = parser.parse_args()
    run(args.clips, args.pipeline)
