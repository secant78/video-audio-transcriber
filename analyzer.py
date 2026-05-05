#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Media Transcript Analyzer
Transcribes audio/video files with Whisper and generates a comprehensive
analysis report using Claude: summary, Q&A, technologies, technical deep-dive,
and performance feedback.

Usage:
    python analyzer.py <media_file> [--model tiny|base|small|medium|large]
    python analyzer.py "recording.mp4"
    python analyzer.py "lecture.mp3" --model medium
"""

import os
import sys
import argparse
import textwrap
from datetime import datetime
from pathlib import Path

import whisper
import anthropic


# ─────────────────────────────────────────────
# Transcription
# ─────────────────────────────────────────────

def transcribe(media_path: str, model_size: str = "base") -> dict:
    """Transcribe a media file using Whisper. Returns the result dict."""
    print(f"\n[1/3] Loading Whisper '{model_size}' model...")
    model = whisper.load_model(model_size)

    print(f"[1/3] Transcribing: {media_path}")
    result = model.transcribe(media_path, verbose=False)
    segments = result.get("segments", [])
    lang = result.get("language", "unknown")
    print(f"      Done - {len(segments)} segments, language: {lang}")
    return result


def build_timestamped_transcript(result: dict) -> str:
    """Format Whisper segments into a readable timestamped transcript."""
    lines = []
    for seg in result["segments"]:
        start = _fmt_time(seg["start"])
        end   = _fmt_time(seg["end"])
        text  = seg["text"].strip()
        lines.append(f"[{start} -> {end}]  {text}")
    return "\n".join(lines)


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ─────────────────────────────────────────────
# Claude analysis
# ─────────────────────────────────────────────

ANALYSIS_PROMPT = (
    "You are an expert technical analyst and communication coach. Below is a "
    "timestamped transcript of a recorded session (lecture, interview, demo, "
    "presentation, or discussion).\n\n"
    "Analyze it thoroughly and produce a structured report with EXACTLY these "
    "seven sections in order, each preceded by its heading line shown below. "
    "Do not omit any section. Be detailed and specific - vague generalities are "
    "not helpful.\n\n"
    "---\n"
    "## 1. DETAILED SUMMARY\n\n"
    "Write a comprehensive summary of the entire session (aim for at least "
    "8-12 sentences). Cover the main narrative arc, key decisions made, "
    "important conclusions, and the overall purpose of the session.\n\n"
    "---\n"
    "## 2. QUESTIONS & ANSWERS\n\n"
    "List every question that was asked (explicitly or implicitly) during the "
    "session and the corresponding answer given. Use this format for each entry:\n\n"
    "**Q:** <question>\n"
    "**A:** <answer>\n"
    "**Timestamp:** <approximate timestamp>\n\n"
    "If a question went unanswered, note that.\n\n"
    "---\n"
    "## 3. TOOLS, TECHNOLOGIES & USE CASES\n\n"
    "For each distinct tool, technology, framework, platform, service, or "
    "concept mentioned, create an entry:\n\n"
    "**Tool/Technology:** <name>\n"
    "**Category:** <e.g., cloud, container, IaC, monitoring, language, database...>\n"
    "**How it was used or discussed in this session:**\n"
    "**General use cases and capabilities:**\n\n"
    "---\n"
    "## 4. TECHNICAL DEEP DIVE\n\n"
    "For each significant technical topic covered in the session, write a "
    "thorough discussion (at least one substantial paragraph per topic). "
    "Explain the underlying concepts, how they connect to each other, potential "
    "pitfalls, best practices, and why the topic matters. Go beyond what was "
    "said in the transcript - add relevant technical context a practitioner "
    "would find valuable.\n\n"
    "---\n"
    "## 5. WHAT WAS DONE WELL\n\n"
    "Identify specific strengths demonstrated by the presenter / participant "
    "in this session - clarity of explanation, good use of examples, accurate "
    "technical content, effective teaching moments, etc. Reference specific "
    "moments or quotes from the transcript to support each point.\n\n"
    "---\n"
    "## 6. AREAS FOR IMPROVEMENT\n\n"
    "Identify specific areas where the presenter / participant could improve. "
    "Be constructive and precise: what was unclear, technically inaccurate, "
    "skipped over, or could have been explained better? Again, reference the "
    "transcript. For each issue, suggest a concrete way to address it.\n\n"
    "---\n"
    "## 7. OVERALL ASSESSMENT\n\n"
    "A final 3-5 sentence evaluation of the session - technical depth, "
    "communication effectiveness, and a recommended priority action for "
    "the next session.\n\n"
    "---\n\n"
    "TRANSCRIPT:\n"
    "{transcript}"
)


def analyze(transcript_text: str, api_key: str) -> str:
    """Send transcript to Claude and return the full analysis markdown."""
    print("[2/3] Sending transcript to Claude for analysis...")

    client = anthropic.Anthropic(api_key=api_key)
    prompt = ANALYSIS_PROMPT.format(transcript=transcript_text)

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )

    print("      Analysis complete.")
    return message.content[0].text


# ─────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────

def build_report(media_path: str, transcript_text: str, analysis: str) -> str:
    """Assemble the final markdown report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    return (
        "# Transcript Analysis Report\n"
        f"**File:** {Path(media_path).name}\n"
        f"**Generated:** {timestamp}\n\n"
        "---\n\n"
        "## FULL TRANSCRIPT\n\n"
        f"{transcript_text}\n\n"
        "---\n\n"
        f"{analysis}\n"
    )


def save_report(report: str, media_path: str) -> str:
    """Write the report to a markdown file next to the source media."""
    stem = Path(media_path).stem
    out_dir = Path(media_path).parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{stem}_analysis_{ts}.md"
    out_path.write_text(report, encoding="utf-8")
    return str(out_path)


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe a media file and generate a full analysis report."
    )
    parser.add_argument(
        "media",
        help="Path to audio or video file (mp3, mp4, wav, m4a, mov, etc.)",
    )
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size. Larger = slower but more accurate. Default: base",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ANTHROPIC_API_KEY"),
        help="Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.",
    )
    parser.add_argument(
        "--transcript-only",
        action="store_true",
        help="Skip Claude analysis and only save the transcript.",
    )
    args = parser.parse_args()

    # Validate inputs
    media_path = args.media.strip('"').strip("'")
    if not Path(media_path).exists():
        print(f"Error: file not found: {media_path}", file=sys.stderr)
        sys.exit(1)

    if not args.transcript_only and not args.api_key:
        print(
            "Error: Anthropic API key required. "
            "Set ANTHROPIC_API_KEY env var or pass --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Step 1 - Transcribe
    result = transcribe(media_path, args.model)
    transcript_text = build_timestamped_transcript(result)

    if args.transcript_only:
        print("\n" + "=" * 60)
        print(transcript_text)
        stem = Path(media_path).stem
        out = Path(media_path).parent / f"{stem}_transcript.txt"
        out.write_text(transcript_text, encoding="utf-8")
        print(f"\nTranscript saved to: {out}")
        return

    # Step 2 - Analyze with Claude
    analysis = analyze(transcript_text, args.api_key)

    # Step 3 - Build & save report
    print("[3/3] Writing report...")
    report = build_report(media_path, transcript_text, analysis)
    out_path = save_report(report, media_path)

    print(f"\nDONE - Report saved to: {out_path}")
    print("\n" + "=" * 60)
    print("PREVIEW - first 80 lines of analysis:")
    print("=" * 60)
    for line in analysis.splitlines()[:80]:
        print(line)


if __name__ == "__main__":
    main()
