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

    # 3. Bitwarden — only attempt if a password was explicitly provided
    if not bw_password or not bw_password.strip():
        return ""

    if not _bw_available():
        return ""

    print("Fetching API key from Bitwarden...")

    # Reuse existing session if it is still valid (avoids a slow re-unlock)
    session = os.environ.get("BW_SESSION", "").strip()
    if session and _bw_session_valid(session):
        print("      Reusing existing vault session.")
    else:
        session = _bw_unlock(bw_password)
        if not session:
            return ""
        os.environ["BW_SESSION"] = session

    # Sync vault to pull any items added/changed since last local cache update
    print("      Syncing vault...")
    try:
        subprocess.run(
            ["bw", "sync", "--nointeraction"],
            env={**os.environ, "BW_SESSION": session},
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as e:
        print(f"      Vault sync warning (continuing): {e}")

    key = _bw_get_notes("DeepSeek API Key", session)
    if key:
        os.environ["DEEPSEEK_API_KEY"] = key
        print("      DeepSeek API key loaded from Bitwarden.")

    # Also fetch Groq key while vault is open
    groq_key = _bw_get_notes("Groq API Key", session)
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        print("      Groq API key loaded from Bitwarden.")

    return key


def _bw_available() -> bool:
    """Check if the Bitwarden CLI is installed."""
    try:
        subprocess.run(["bw", "--version"], capture_output=True, check=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def _bw_session_valid(session: str) -> bool:
    """Return True if the given session token can already reach the vault."""
    if not session:
        return False
    try:
        result = subprocess.run(
            ["bw", "status", "--nointeraction"],
            env={**os.environ, "BW_SESSION": session},
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Handle both compact and spaced JSON: "status":"unlocked" or "status": "unlocked"
        return result.returncode == 0 and "unlocked" in result.stdout
    except Exception:
        return False


def _bw_unlock(password: str = "") -> str:
    """Unlock Bitwarden vault and return a session token.
    Uses --offline so it never syncs with the server (much faster).
    Requires a non-empty password — never opens an interactive terminal prompt.
    """
    if not password or not password.strip():
        return ""
    try:
        result = subprocess.run(
            ["bw", "unlock", "--raw", "--passwordenv", "BW_PASSWORD",
             "--nointeraction"],
            env={**os.environ, "BW_PASSWORD": password.strip()},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            print(f"      bw unlock failed (exit {result.returncode}): {result.stderr.strip()[:200]}")
        return result.stdout.strip() if result.returncode == 0 else ""
    except subprocess.TimeoutExpired:
        print("      Bitwarden unlock timed out after 60s.")
        return ""
    except Exception:
        return ""


def _bw_get_notes(item_name: str, session: str) -> str:
    """Retrieve the notes field of a Bitwarden item."""
    try:
        result = subprocess.run(
            ["bw", "get", "notes", item_name, "--nointeraction"],
            env={**os.environ, "BW_SESSION": session},
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            print(f"      bw get notes '{item_name}' failed (exit {result.returncode}): {result.stderr.strip()[:200]}")
        return result.stdout.strip() if result.returncode == 0 else ""
    except subprocess.TimeoutExpired:
        print(f"      Bitwarden get '{item_name}' timed out after 15s.")
        return ""
    except Exception as e:
        print(f"      Bitwarden get '{item_name}' exception: {e}")
        return ""


# ─────────────────────────────────────────────
# Groq transcription (cloud, fast, free tier)
# ─────────────────────────────────────────────

# At 128kbps MP3: 25 MB = ~1600 seconds. Use 1440s (24 min) per chunk to stay safely under.
GROQ_CHUNK_DURATION = 1440
GROQ_MODEL = "whisper-large-v3"


def get_audio_duration(path: str) -> float:
    """Return duration in seconds using ffmpeg."""
    import re
    result = subprocess.run(
        ["ffmpeg", "-i", path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", result.stderr)
    if match:
        h, m, s = match.groups()
        return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


def split_audio(mp3_path: str, chunk_duration: int, tmp_dir: Path) -> list:
    """Split an MP3 into fixed-duration chunks. Returns list of chunk paths."""
    chunk_pattern = str(tmp_dir / "chunk_%03d.mp3")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", mp3_path,
            "-f", "segment",
            "-segment_time", str(chunk_duration),
            "-c", "copy",
            "-reset_timestamps", "1",
            chunk_pattern,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return sorted(tmp_dir.glob("chunk_*.mp3"))


def transcribe_with_groq(
    media_path: str,
    groq_api_key: str,
    language: str = "en",
    progress_callback=None,
) -> dict:
    """
    Convert media to MP3, split into 25 MB chunks, transcribe each with
    Groq whisper-large-v3, and combine into a single timestamped transcript.
    """
    from groq import Groq

    client = Groq(api_key=groq_api_key)
    tmp_dir = Path(tempfile.mkdtemp())

    def _progress(pct, desc):
        if progress_callback:
            progress_callback(pct, desc)

    try:
        # Step A — Convert to MP3
        _progress(0.0, "Converting to MP3...")
        mp3_path = tmp_dir / "audio.mp3"
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", media_path, "-vn", "-ab", "128k", str(mp3_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0 or not mp3_path.exists():
            raise RuntimeError("ffmpeg conversion failed.")
        _progress(0.1, "Conversion complete. Splitting into chunks...")

        # Step B — Split
        total_duration = get_audio_duration(str(mp3_path))
        chunks = split_audio(str(mp3_path), GROQ_CHUNK_DURATION, tmp_dir)
        n = len(chunks)
        print(f"      Split into {n} chunk(s) of up to {GROQ_CHUNK_DURATION}s each.")
        _progress(0.15, f"Split into {n} chunk(s). Starting transcription...")

        # Step C — Transcribe each chunk
        all_segments = []
        time_offset = 0.0

        for i, chunk_path in enumerate(chunks):
            chunk_label = f"[Chunk {i + 1}/{n}]"
            pct = 0.15 + (i / n) * 0.75
            _progress(pct, f"Transcribing {chunk_label}...")
            print(f"      Transcribing {chunk_label}: {chunk_path.name}")

            with open(chunk_path, "rb") as f:
                response = client.audio.transcriptions.create(
                    file=(chunk_path.name, f.read()),
                    model=GROQ_MODEL,
                    language=language,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )

            chunk_duration = get_audio_duration(str(chunk_path))

            for seg in response.segments:
                all_segments.append({
                    "start": seg.start + time_offset,
                    "end":   seg.end + time_offset,
                    "text":  seg.text.strip(),
                })

            time_offset += chunk_duration

        _progress(0.95, "Combining transcripts...")
        print(f"      Done — {len(all_segments)} total segments across {n} chunk(s).")

        return {"segments": all_segments, "language": language}

    finally:
        # Clean up all temp files
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ─────────────────────────────────────────────
# Local Whisper transcription
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
    try:
        import torch
        if torch.cuda.is_available():
            print("      GPU detected - using CUDA (float16)")
            return "cuda", "float16"
    except Exception:
        pass
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
