#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transcript Analyzer - Web UI
Run with: python ui.py
Then open http://localhost:7860 in your browser.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import gradio as gr
import torch
from faster_whisper import WhisperModel
from openai import OpenAI

from analyzer import (
    convert_to_mp3,
    get_device,
    get_api_key,
    build_report,
    save_report,
    ANALYSIS_PROMPT,
    _fmt_time,
)


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

def process_media(file_path, model_size, language, bw_password, progress=gr.Progress()):
    """Full pipeline: transcribe + analyze with per-step resetting progress bars."""

    if not file_path:
        raise gr.Error("Please upload a file first.")

    api_key = get_api_key(bw_password or "")
    if not api_key:
        raise gr.Error(
            "DeepSeek API key not found. "
            "Store it in Bitwarden as 'DeepSeek API Key' or set DEEPSEEK_API_KEY."
        )

    media_path = file_path
    temp_mp3 = None

    # ── Step 1: Convert to MP3 ───────────────────
    ext = Path(media_path).suffix.lower()
    if ext in (".webm", ".mkv", ".mov", ".mp4"):
        progress(0.0, desc="[Step 1/3] Starting MP3 conversion...")
        progress(0.4, desc="[Step 1/3] Converting video to audio...")
        converted = convert_to_mp3(media_path)
        if converted != media_path:
            temp_mp3 = converted
            media_path = converted
        progress(1.0, desc="[Step 1/3] Conversion complete!")
        progress(0.0, desc="")   # reset for next step

    # ── Step 2: Transcribe ───────────────────────
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
    progress(0.0, desc="")   # reset for next step

    # ── Step 3: DeepSeek analysis ────────────────
    progress(0.0, desc="[Step 3/3] Connecting to DeepSeek V3...")
    analysis = analyze_streaming(transcript_text, api_key, progress, "[Step 3/3] Generating report...")
    progress(1.0, desc="[Step 3/3] Analysis complete!")

    # Build & save report
    report = build_report(media_path, transcript_text, analysis)
    out_path = save_report(report, file_path)

    return transcript_text, analysis, report, out_path


def process_transcript(transcript_text, bw_password, progress=gr.Progress()):
    """Analyze an existing transcript with per-step resetting progress bars."""

    if not transcript_text or not transcript_text.strip():
        raise gr.Error("Please paste or upload a transcript first.")

    api_key = get_api_key(bw_password or "")
    if not api_key:
        raise gr.Error(
            "DeepSeek API key not found. "
            "Store it in Bitwarden as 'DeepSeek API Key' or set DEEPSEEK_API_KEY."
        )

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

    with gr.Row():
        bw_password = gr.Textbox(
            label="Bitwarden Master Password",
            type="password",
            placeholder="Enter only if prompted for API key...",
            scale=1,
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
                    model_choice = gr.Dropdown(
                        choices=["tiny", "base", "small", "medium", "large"],
                        value="medium",
                        label="Whisper Model",
                        info="medium = best balance of speed and accuracy",
                    )
                    language_choice = gr.Textbox(
                        value="en",
                        label="Language Code",
                        info="en, es, fr, de, etc.",
                        max_lines=1,
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
                inputs=[media_file, model_choice, language_choice, bw_password],
                outputs=[media_transcript_out, media_analysis_out, media_report_out, media_save_path],
            )

        # ── Tab 2: Paste existing transcript ────────────
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
                inputs=[transcript_input, bw_password],
                outputs=[transcript_analysis_out, transcript_report_out, transcript_save_path],
            )

if __name__ == "__main__":
    app.launch(inbrowser=True, theme=gr.themes.Soft(), css=CSS)
