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
import warnings
import argparse
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

# Suppress noisy HuggingFace/symlink warnings on Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

import torch
from faster_whisper import WhisperModel
from openai import OpenAI


# ─────────────────────────────────────────────
# Bitwarden credential retrieval
# ─────────────────────────────────────────────

def get_api_key(bw_password: str = "") -> str:
    """
    Return the DeepSeek API key. Priority order:
      1. DEEPSEEK_API_KEY env var (already set)
      2. .env file in the project directory
      3. Bitwarden vault (uses bw_password if provided, else prompts terminal)
    """
    # 1. Already in environment
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key

    # 2. .env file next to this script
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("DEEPSEEK_API_KEY="):
                key = line.split("=", 1)[1].strip()
                if key:
                    os.environ["DEEPSEEK_API_KEY"] = key
                    return key

    # 3. Bitwarden
    if not _bw_available():
        return ""

    print("Fetching API key from Bitwarden...")

    session = os.environ.get("BW_SESSION", "").strip()
    if not session:
        session = _bw_unlock(bw_password)
        if not session:
            return ""
        os.environ["BW_SESSION"] = session

    key = _bw_get_notes("DeepSeek API Key", session)
    if key:
        os.environ["DEEPSEEK_API_KEY"] = key
        print("      API key loaded from Bitwarden.")
    return key


def _bw_available() -> bool:
    """Check if the Bitwarden CLI is installed."""
    try:
        subprocess.run(["bw", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _bw_unlock(password: str = "") -> str:
    """Unlock Bitwarden vault and return a session token.
    If password is provided it is passed via stdin (no terminal prompt).
    Otherwise the terminal prompt is shown as usual.
    """
    try:
        if password:
            result = subprocess.run(
                ["bw", "unlock", "--raw", "--passwordenv", "BW_PASSWORD"],
                env={**os.environ, "BW_PASSWORD": password},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        else:
            result = subprocess.run(
                ["bw", "unlock", "--raw"],
                stdout=subprocess.PIPE,
                text=True,
            )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _bw_get_notes(item_name: str, session: str) -> str:
    """Retrieve the notes field of a Bitwarden item."""
    try:
        result = subprocess.run(
            ["bw", "get", "notes", item_name, "--session", session],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


# ─────────────────────────────────────────────
# Transcription
# ─────────────────────────────────────────────

def convert_to_mp3(media_path: str) -> str:
    """Convert WebM (or any format) to MP3 using ffmpeg to avoid decoder issues."""
    out_path = Path(tempfile.mktemp(suffix=".mp3"))
    print(f"      Converting to MP3 for reliable decoding...")
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", media_path, "-vn", "-ab", "128k", str(out_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        print("      Warning: ffmpeg conversion failed, using original file.")
        return media_path
    print(f"      Converted: {out_path}")
    return str(out_path)


def get_device() -> tuple:
    """Detect GPU availability and return (device, compute_type)."""
    if torch.cuda.is_available():
        print("      GPU detected - using CUDA (float16)")
        return "cuda", "float16"
    else:
        print("      No GPU detected - using CPU (int8, optimized)")
        return "cpu", "int8"


def transcribe(media_path: str, model_size: str = "medium", language: str = "en") -> dict:
    """Transcribe a media file using faster-whisper. Returns a segments dict."""
    # Fix 1: Convert WebM files to avoid variable bitrate decoder issues
    ext = Path(media_path).suffix.lower()
    temp_file = None
    if ext in (".webm", ".mkv", ".mov", ".mp4"):
        converted = convert_to_mp3(media_path)
        if converted != media_path:
            temp_file = converted
            media_path = converted

    # Fix 2: Auto-detect GPU vs CPU with int8 quantization on CPU for speed
    device, compute_type = get_device()

    print(f"\n[1/3] Loading faster-whisper '{model_size}' model...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # Fix 3: Force language to skip auto-detection errors
    print(f"[1/3] Transcribing ({language.upper()}, device={device}, compute={compute_type})...")
    segments_gen, info = model.transcribe(media_path, language=language, beam_size=5)

    # Consume the generator into a list
    segments = []
    for seg in segments_gen:
        segments.append({
            "start": seg.start,
            "end":   seg.end,
            "text":  seg.text.strip(),
        })

    print(f"      Done - {len(segments)} segments, detected language: {info.language}")

    # Clean up temp file if we created one
    if temp_file and Path(temp_file).exists():
        Path(temp_file).unlink()

    return {"segments": segments, "language": info.language}


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
    """Send transcript to DeepSeek V3 and return the full analysis markdown."""
    print("[2/3] Sending transcript to DeepSeek V3 for analysis...")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )
    prompt = ANALYSIS_PROMPT.format(transcript=transcript_text)

    response = client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )

    print("      Analysis complete.")
    return response.choices[0].message.content


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
        nargs="?",
        help="Path to audio or video file (mp3, mp4, wav, m4a, mov, etc.)",
    )
    parser.add_argument(
        "--transcript-file",
        help="Path to an existing transcript text file. Skips Whisper entirely.",
    )
    parser.add_argument(
        "--model",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size. Larger = slower but more accurate. Default: medium",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language of the audio (e.g. en, es, fr). Skips auto-detection. Default: en",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="DeepSeek API key. Falls back to Bitwarden or DEEPSEEK_API_KEY env var.",
    )
    parser.add_argument(
        "--transcript-only",
        action="store_true",
        help="Skip DeepSeek analysis and only save the transcript.",
    )
    args = parser.parse_args()

    # Validate inputs
    if not args.transcript_file and not args.media:
        print("Error: provide a media file or --transcript-file.", file=sys.stderr)
        sys.exit(1)

    # Resolve API key: CLI flag > Bitwarden > env var
    api_key = args.api_key or get_api_key()

    if not args.transcript_only and not api_key:
        print(
            "Error: DeepSeek API key not found. "
            "Store it in Bitwarden as 'DeepSeek API Key', "
            "set DEEPSEEK_API_KEY env var, or pass --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Step 1 - Get transcript (from file or by transcribing media)
    if args.transcript_file:
        transcript_path = args.transcript_file.strip('"').strip("'")
        if not Path(transcript_path).exists():
            print(f"Error: transcript file not found: {transcript_path}", file=sys.stderr)
            sys.exit(1)
        print(f"\n[1/3] Loading transcript from: {transcript_path}")
        transcript_text = Path(transcript_path).read_text(encoding="utf-8")
        media_path = transcript_path  # used for naming the output file
        print(f"      Done - {len(transcript_text.splitlines())} lines loaded.")
    else:
        media_path = args.media.strip('"').strip("'")
        if not Path(media_path).exists():
            print(f"Error: file not found: {media_path}", file=sys.stderr)
            sys.exit(1)
        result = transcribe(media_path, args.model, args.language)
        transcript_text = build_timestamped_transcript(result)

    if args.transcript_only:
        print("\n" + "=" * 60)
        print(transcript_text)
        stem = Path(media_path).stem
        out = Path(media_path).parent / f"{stem}_transcript.txt"
        out.write_text(transcript_text, encoding="utf-8")
        print(f"\nTranscript saved to: {out}")
        return

    # Step 2 - Analyze with DeepSeek
    analysis = analyze(transcript_text, api_key)

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
