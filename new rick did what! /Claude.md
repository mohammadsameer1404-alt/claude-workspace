# Rick Tedwatt — YouTube Shorts Pipeline

Channel goal: 10M views / 1000 subscribers in 15–20 days posting daily Rick and Morty Shorts (US audience).

## Pipeline Overview

```
[Stage 1 — Research]   rick-and-morty-fan skill  →  clips_spec.json
[Stage 2 — Edit]       rick-and-morty-shorts skill → output/raw/*.mp4
[Stage 3 — Audit]      shorts-auditor skill        → output/passed/*.mp4
[Stage 4 — Post]       Manual review → upload to YouTube
```

## Quick Start

```bash
# Stage 1: Run the research skill in Claude to generate a spec file
# /rick-and-morty-fan → saves clips_spec.json

# Stage 2+3: Edit and audit in one command
python3 run_pipeline.py clips_spec.json

# Audit only (if you already have clips in output/raw/)
python3 run_pipeline.py dummy.json --audit-only

# Custom hook title and threshold
python3 run_pipeline.py clips_spec.json --hook "Rick Breaks Physics Again" --threshold 8.6
```

## Paths

| Resource | Path |
|---|---|
| Episode library | `/Users/sameermohammad/Movies/Rick and Morty` |
| Output (raw clips) | `output/raw/` |
| Output (passed clips) | `output/passed/` |
| Music library | `~/.claude/skills/rick-and-morty-shorts/music/` |
| Reject log | `~/.claude/skills/shorts-auditor/reject-log.jsonl` |

## API Keys Required

Add to `~/.zshrc` and run `source ~/.zshrc`:

```bash
export GEMINI_API_KEY="..."      # ✅ already set — console.cloud.google.com
export GROQ_API_KEY="gsk_..."    # ⚠️  needed — console.groq.com → API Keys (free)
export OPENROUTER_API_KEY="..."  # ⚠️  needed — openrouter.ai → Keys (free, no card)
```

## Audit Score Meaning

Passing clips are saved as `8.7_rm_s01e01_001.mp4` (score prefix = worst of the 3 AIs).
Only clips where **all 3 AIs score ≥ 8.6** are saved. This is intentionally strict.

| AI | Model | API |
|---|---|---|
| Gemini | gemini-2.5-flash | Google AI Studio (free) |
| Qwen2.5-VL | qwen2.5-vl-72b-instruct:free | OpenRouter (free) |
| Llama Vision | llama-3.2-90b-vision-preview | Groq (free) |

## Music Setup

Music files go in `~/.claude/skills/rick-and-morty-shorts/music/` named:
```
energy_high_synthwave_space_chase.mp3
energy_medium_cosmic_drift.mp3
energy_low_ambient_portal_hum.mp3
energy_comedy_quirky_blip.mp3
```
Download royalty-free tracks from pixabay.com/music (no attribution needed).

## Season Folder Naming

Episodes are at `/Users/sameermohammad/Movies/Rick and Morty/`:
- Seasons 1–4: `season 1/`, `season 2/`, ...
- Seasons 5–8: `Season 5/`, `Season 6/`, ...
