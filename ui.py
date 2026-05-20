#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transcript Analyzer - Web UI
Run with: python ui.py
Then open http://localhost:7860 in your browser.
"""

import os
import re
import tempfile
import subprocess
from datetime import datetime
from pathlib import Path

import gradio as gr
from faster_whisper import WhisperModel
from openai import OpenAI

from analyzer import (
    convert_to_mp3,
    get_device,
    get_api_key,
    build_report,
    save_report,
    build_timestamped_transcript,
    transcribe_with_groq,
    ANALYSIS_PROMPT,
    _fmt_time,
)


def get_media_duration(media_path: str) -> float:
    """Return duration in seconds using ffmpeg."""
    import re
    result = subprocess.run(
        ["ffmpeg", "-i", media_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", result.stderr)
    if match:
        h, m, s = match.groups()
        return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


def convert_with_progress(media_path: str, output_path: str, progress, label: str) -> bool:
    """Run ffmpeg and stream real-time progress updates."""
    duration = get_media_duration(media_path)

    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-i", media_path, "-vn", "-ab", "128k",
            str(output_path), "-progress", "pipe:1", "-nostats",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    for line in proc.stdout:
        line = line.strip()
        if line.startswith("out_time_ms="):
            try:
                ms = int(line.split("=")[1])
                elapsed = ms / 1_000_000
                pct = min(elapsed / duration, 0.99) if duration else 0.5
                progress(pct, desc=f"{label} {int(pct * 100)}%")
            except ValueError:
                pass

    proc.wait()
    return proc.returncode == 0


def analyze_streaming(transcript_text: str, api_key: str, progress, base_desc: str) -> str:
    """Call DeepSeek with streaming so progress updates token-by-token."""
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    prompt = ANALYSIS_PROMPT.format(transcript=transcript_text)
    full_text = ""
    chunk_count = 0
    est_total = 500   # rough expected chunks; scales progress to ~99%

    with client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full_text += delta
            chunk_count += 1
            pct = min(chunk_count / est_total, 0.99)
            progress(pct, desc=f"{base_desc} {int(pct * 100)}%")

    return full_text


# ─────────────────────────────────────────────
# Core processing
# ─────────────────────────────────────────────

def process_media(file_path, transcription_method, model_size, language, progress=gr.Progress()):
    """Full pipeline: transcribe + analyze with per-step resetting progress bars."""

    if not file_path:
        raise gr.Error("Please upload a file first.")

    api_key = get_api_key()
    if not api_key:
        raise gr.Error("DeepSeek API key not found. Enter your Bitwarden master password above and click Unlock Vault.")

    media_path = file_path

    # ── Groq path ─────────────────────────────────
    if transcription_method == "Groq (Cloud - Fast, Free)":
        groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not groq_api_key:
            raise gr.Error("Groq API key not found. Unlock your Bitwarden vault first.")

        progress(0.0, desc="[Step 1/2] Starting Groq transcription...")

        def groq_progress(pct, desc):
            progress(pct, desc=f"[Step 1/2] {desc}")

        result = transcribe_with_groq(
            media_path, groq_api_key.strip(), language, groq_progress
        )
        transcript_text = build_timestamped_transcript(result)
        progress(1.0, desc="[Step 1/2] Transcription complete!")
        progress(0.0, desc="")

        progress(0.0, desc="[Step 2/2] Connecting to DeepSeek V3...")
        analysis = analyze_streaming(transcript_text, api_key, progress, "[Step 2/2] Generating report...")
        progress(1.0, desc="[Step 2/2] Analysis complete!")

    # ── Local Whisper path ───────────────────────
    else:
        temp_mp3 = None
        ext = Path(media_path).suffix.lower()
        if ext in (".webm", ".mkv", ".mov", ".mp4"):
            progress(0.0, desc="[Step 1/3] Starting MP3 conversion...")
            out_mp3 = Path(tempfile.mktemp(suffix=".mp3"))
            success = convert_with_progress(media_path, out_mp3, progress, "[Step 1/3] Converting...")
            if success and out_mp3.exists():
                temp_mp3 = str(out_mp3)
                media_path = temp_mp3
            progress(1.0, desc="[Step 1/3] Conversion complete!")
            progress(0.0, desc="")

        progress(0.0, desc=f"[Step 2/3] Loading Whisper '{model_size}' model...")
        device, compute_type = get_device()
        model = WhisperModel(model_size, device=device, compute_type=compute_type)

        progress(0.02, desc="[Step 2/3] Transcribing audio...")
        segments_gen, info = model.transcribe(media_path, language=language, beam_size=5)

        segments = []
        transcript_lines = []
        total_duration = info.duration if info.duration else 1

        for seg in segments_gen:
            segments.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
            transcript_lines.append(
                f"[{_fmt_time(seg.start)} -> {_fmt_time(seg.end)}]  {seg.text.strip()}"
            )
            pct = seg.end / total_duration
            progress(pct, desc=f"[Step 2/3] Transcribing... {pct:.0%} complete")

        transcript_text = "\n".join(transcript_lines)

        if temp_mp3 and Path(temp_mp3).exists():
            Path(temp_mp3).unlink()

        progress(1.0, desc="[Step 2/3] Transcription complete!")
        progress(0.0, desc="")

        progress(0.0, desc="[Step 3/3] Connecting to DeepSeek V3...")
        analysis = analyze_streaming(transcript_text, api_key, progress, "[Step 3/3] Generating report...")
        progress(1.0, desc="[Step 3/3] Analysis complete!")

    # Build & save report to Downloads folder (Gradio temp path is not reliable)
    stem = Path(file_path).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path.home() / "Downloads" / f"{stem}_analysis_{ts}.md"
    report = build_report(file_path, transcript_text, analysis)
    out_path.write_text(report, encoding="utf-8")

    return transcript_text, analysis, report, str(out_path)


def extract_audio_only(file_path, output_format, progress=gr.Progress()):
    """Extract audio from a video file and save to Downloads."""
    if not file_path:
        raise gr.Error("Please upload a video file first.")

    progress(0.0, desc="Starting audio extraction...")
    src = Path(file_path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path.home() / "Downloads" / f"{src.stem}_audio_{ts}.{output_format}"

    progress(0.3, desc=f"Extracting audio as {output_format.upper()}...")

    extra = ["-ab", "128k"] if output_format == "mp3" else []
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", file_path, "-vn"] + extra + [str(out_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if result.returncode != 0:
        raise gr.Error("Audio extraction failed. Make sure the file is a valid video.")

    progress(1.0, desc="Extraction complete!")
    return str(out_path), gr.update(value=str(out_path), visible=True)


def process_transcript(transcript_text, progress=gr.Progress()):
    """Analyze an existing transcript with per-step resetting progress bars."""

    if not transcript_text or not transcript_text.strip():
        raise gr.Error("Please paste or upload a transcript first.")

    api_key = get_api_key()
    if not api_key:
        raise gr.Error("DeepSeek API key not found. Enter your Bitwarden master password above and click Unlock Vault.")

    # ── Step 1: DeepSeek analysis ────────────────
    progress(0.0, desc="[Step 1/2] Connecting to DeepSeek V3...")
    analysis = analyze_streaming(transcript_text, api_key, progress, "[Step 1/2] Generating report...")
    progress(1.0, desc="[Step 1/2] Analysis complete!")
    progress(0.0, desc="")   # reset for next step

    # ── Step 2: Build & save report ──────────────
    progress(0.5, desc="[Step 2/2] Building report...")
    report = (
        "# Transcript Analysis Report\n"
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        "---\n\n"
        "## FULL TRANSCRIPT\n\n"
        f"{transcript_text}\n\n"
        "---\n\n"
        f"{analysis}\n"
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path.home() / "Downloads" / f"transcript_analysis_{ts}.md"
    out_path.write_text(report, encoding="utf-8")
    progress(1.0, desc="[Step 2/2] Report saved!")

    return analysis, report, str(out_path)


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

CSS = """
#title { text-align: center; margin-bottom: 8px; }
#subtitle { text-align: center; color: #6b7280; margin-bottom: 24px; }
.tab-nav button { font-size: 15px; }
#run-btn { background: #2563eb; color: white; font-size: 16px; }
#run-btn:hover { background: #1d4ed8; }
"""

with gr.Blocks(title="Transcript Analyzer") as app:

    gr.Markdown("# Transcript Analyzer", elem_id="title")
    gr.Markdown(
        "Transcribe audio/video with Whisper · Analyze with DeepSeek V3",
        elem_id="subtitle",
    )

    # ── Vault unlock row ────────────────────────
    vault_status = gr.Textbox(
        value="",
        label="Vault Status",
        interactive=False,
        visible=False,
    )

    with gr.Row(visible=False) as vault_row:
        bw_password = gr.Textbox(
            label="Bitwarden Master Password",
            type="password",
            placeholder="Enter your Bitwarden master password...",
            scale=3,
        )
        unlock_btn = gr.Button("Unlock Vault", variant="primary", scale=1)

    def unlock_vault(password):
        """Fetch all API keys from Bitwarden and populate fields."""
        if not password or not password.strip():
            return (
                gr.update(value="Please enter your Bitwarden master password.", visible=True),
                gr.update(visible=True),
                gr.update(),
            )
        key = get_api_key(password.strip())
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if key:
            return (
                gr.update(value="Vault unlocked. Ready to go.", visible=True),
                gr.update(visible=False),
                gr.update(value=groq_key),
            )
        return (
            gr.update(value="Wrong password or vault locked. Try again.", visible=True),
            gr.update(visible=True),
            gr.update(),
        )

    with gr.Tabs():

        # ── Tab 1: Upload media ──────────────────────────
        with gr.TabItem("Upload Audio / Video"):

            with gr.Row():
                with gr.Column(scale=2):
                    media_file = gr.File(
                        label="Drop or click to upload (mp3, mp4, webm, wav, m4a, mov...)",
                        file_types=["audio", "video"],
                    )
                with gr.Column(scale=1):
                    transcription_method = gr.Radio(
                        choices=["Local Whisper (CPU)", "Groq (Cloud - Fast, Free)"],
                        value="Groq (Cloud - Fast, Free)",
                        label="Transcription Method",
                    )
                    model_choice = gr.Dropdown(
                        choices=["tiny", "base", "small", "medium", "large"],
                        value="medium",
                        label="Whisper Model (Local only)",
                        info="Only used when Local Whisper is selected",
                        visible=False,
                    )
                    language_choice = gr.Textbox(
                        value="en",
                        label="Language Code",
                        info="en, es, fr, de, etc.",
                        max_lines=1,
                    )

            def toggle_method(method):
                is_local = method == "Local Whisper (CPU)"
                return gr.update(visible=is_local)

            transcription_method.change(
                fn=toggle_method,
                inputs=transcription_method,
                outputs=model_choice,
            )

            run_media_btn = gr.Button("Transcribe & Analyze", variant="primary", elem_id="run-btn")

            with gr.Tabs():
                with gr.TabItem("Transcript"):
                    media_transcript_out = gr.Textbox(
                        label="Full Transcript",
                        lines=20,

                        interactive=False,
                    )
                with gr.TabItem("Analysis"):
                    media_analysis_out = gr.Markdown(label="Analysis Report")
                with gr.TabItem("Full Report"):
                    media_report_out = gr.Textbox(
                        label="Full Report (markdown)",
                        lines=20,

                        interactive=False,
                    )

            media_save_path = gr.Textbox(label="Report saved to", interactive=False)

            run_media_btn.click(
                fn=process_media,
                inputs=[media_file, transcription_method, groq_api_key_input, model_choice, language_choice],
                outputs=[media_transcript_out, media_analysis_out, media_report_out, media_save_path],
            )

        # ── Tab 2: Extract audio only ────────────────────
        with gr.TabItem("Extract Audio from Video"):

            with gr.Row():
                with gr.Column(scale=2):
                    audio_video_file = gr.File(
                        label="Drop or click to upload a video (mp4, webm, mov, mkv...)",
                        file_types=["video"],
                    )
                with gr.Column(scale=1):
                    audio_format = gr.Dropdown(
                        choices=["mp3", "wav", "m4a", "flac"],
                        value="mp3",
                        label="Output Format",
                        info="mp3 = smallest file size, wav = highest quality",
                    )

            extract_btn = gr.Button("Extract Audio", variant="primary")
            audio_save_path = gr.Textbox(label="Audio saved to", interactive=False, visible=False)

            extract_btn.click(
                fn=extract_audio_only,
                inputs=[audio_video_file, audio_format],
                outputs=[audio_save_path, audio_save_path],
            )

        # ── Tab 3: Paste existing transcript ─────────────
        with gr.TabItem("Analyze Existing Transcript"):

            with gr.Row():
                with gr.Column():
                    transcript_input = gr.Textbox(
                        label="Paste your transcript here",
                        lines=15,
                        placeholder="Paste transcript text here, or upload a .txt file below...",
                    )
                    transcript_file_upload = gr.File(
                        label="Or upload a .txt transcript file",
                        file_types=[".txt"],
                    )

            def load_transcript_file(f):
                if f is None:
                    return ""
                return Path(f).read_text(encoding="utf-8")

            transcript_file_upload.change(
                fn=load_transcript_file,
                inputs=transcript_file_upload,
                outputs=transcript_input,
            )

            run_transcript_btn = gr.Button("Analyze Transcript", variant="primary", elem_id="run-btn")

            with gr.Tabs():
                with gr.TabItem("Analysis"):
                    transcript_analysis_out = gr.Markdown(label="Analysis Report")
                with gr.TabItem("Full Report"):
                    transcript_report_out = gr.Textbox(
                        label="Full Report (markdown)",
                        lines=20,

                        interactive=False,
                    )

            transcript_save_path = gr.Textbox(label="Report saved to", interactive=False)

            run_transcript_btn.click(
                fn=process_transcript,
                inputs=transcript_input,
                outputs=[transcript_analysis_out, transcript_report_out, transcript_save_path],
            )

    unlock_btn.click(
        fn=lambda: gr.update(value="Connecting to Bitwarden vault...", visible=True),
        inputs=None,
        outputs=vault_status,
    ).then(
        fn=unlock_vault,
        inputs=bw_password,
        outputs=[vault_status, vault_row, groq_api_key_input],
    )

    def check_vault_on_load():
        """On startup, only check env/file — never attempt Bitwarden unlock."""
        key = os.environ.get("DEEPSEEK_API_KEY", "").strip()

        if not key:
            env_file = Path(__file__).parent / ".env"
            if env_file.exists():
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("DEEPSEEK_API_KEY="):
                        key = line.split("=", 1)[1].strip()
                        break

        if key:
            os.environ["DEEPSEEK_API_KEY"] = key
            return (
                gr.update(visible=False),
                gr.update(value="API key loaded. Ready to go.", visible=True),
            )

        return (
            gr.update(visible=True),
            gr.update(value="Enter your Bitwarden master password and click Unlock Vault.", visible=True),
        )

    app.load(fn=check_vault_on_load, outputs=[vault_row, vault_status])

if __name__ == "__main__":
    app.queue()
    app.launch(inbrowser=True, theme=gr.themes.Soft(), css=CSS)
