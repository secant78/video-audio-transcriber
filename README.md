# Video/Audio Transcript Analyzer

Transcribes recorded audio or video files and generates a comprehensive analysis report. Uses **faster-whisper** (local, free) for transcription and **DeepSeek V3** for AI analysis. Includes a **Gradio web UI** with per-step progress bars.

## What It Does

Given any audio or video file (or an existing transcript), the program produces a structured markdown report with:

1. **Full Transcript** - timestamped transcription of the entire recording
2. **Detailed Summary** - 8-12 sentence narrative overview of the session
3. **Questions & Answers** - every Q&A pair extracted with timestamps
4. **Tools, Technologies & Use Cases** - every tool/tech mentioned, its category, how it was used, and general capabilities
5. **Technical Deep Dive** - in-depth discussion of every major technical topic covered
6. **What Was Done Well** - specific strengths with references to the transcript
7. **Areas for Improvement** - constructive, actionable feedback with suggestions
8. **Overall Assessment** - final evaluation and recommended priority action

---

## Running the Web UI

```bash
python ui.py
```

Opens **http://localhost:7860** in your browser automatically.

### UI Features

**Tab 1 — Upload Audio / Video**
- Drag and drop any audio or video file
- Select Whisper model size and language
- Click **Transcribe & Analyze**
- Per-step progress bars that reset between each stage:
  - Step 1/3: MP3 conversion
  - Step 2/3: Whisper transcription (live % per audio segment)
  - Step 3/3: DeepSeek report generation (live token streaming)
- Results shown in three sub-tabs: Transcript, Analysis, Full Report
- Report auto-saved with path displayed at the bottom

**Tab 2 — Analyze Existing Transcript**
- Paste transcript text directly or upload a `.txt` file
- Click **Analyze Transcript** — skips Whisper, goes straight to DeepSeek
- Per-step progress bars:
  - Step 1/2: DeepSeek report generation (live token streaming)
  - Step 2/2: Building and saving report
- Report auto-saved to your Downloads folder

**Bitwarden Password Field**
- Enter your Bitwarden master password in the UI — no terminal prompts needed
- API key is fetched from your Bitwarden vault automatically at runtime

---

## Running from the Command Line

```bash
# Full pipeline: transcribe a media file + generate analysis
python analyzer.py "recording.mp3"

# Transcribe a video file (auto-converts to MP3 first)
python analyzer.py "lecture.mp4"

# Use a transcript you already have (skips Whisper entirely)
python analyzer.py --transcript-file "transcript.txt"

# Only transcribe, skip DeepSeek analysis
python analyzer.py "recording.mp3" --transcript-only

# Specify language (default: English)
python analyzer.py "recording.mp3" --language es

# Pass API key directly
python analyzer.py "recording.mp3" --api-key sk-...
```

---

## Prerequisites

**Python 3.8+** and **ffmpeg** must be installed.

```bash
# Windows
winget install ffmpeg

# Mac
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

You will also need a **DeepSeek API key** from [platform.deepseek.com](https://platform.deepseek.com).

---

## API Key Setup (Bitwarden)

The app retrieves your DeepSeek API key from Bitwarden at runtime — the key is never saved to disk.

**Store the key in Bitwarden (one time):**

```powershell
# Log in and unlock
bw login
$env:BW_SESSION=$(bw unlock --raw)

# Save the key as a secure note
$json = '{"type":2,"name":"DeepSeek API Key","notes":"sk-YOUR-KEY-HERE","secureNote":{"type":0},"favorite":false,"folderId":null}'
$encoded = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($json))
bw create item $encoded
```

**Using the UI:** Enter your Bitwarden master password in the password field at the top of the page before clicking any button.

**Using the CLI:** The app prompts for your master password in the terminal automatically.

---

## Whisper Model Sizes

Whisper runs locally and is completely free. Larger models are slower but more accurate.

| Model | Download Size | Speed | Best For |
|-------|-------------|-------|----------|
| `tiny` | 39 MB | Fastest | Quick drafts |
| `base` | 74 MB | Fast | Good balance |
| `small` | 244 MB | Moderate | Better accuracy |
| `medium` | 769 MB | Slow | Technical/dense audio (default) |
| `large` | 1.5 GB | Slowest | Maximum accuracy |

Models are downloaded automatically on first use and cached locally. Uses **int8 quantization on CPU** for ~4x faster transcription vs standard Whisper.

---

## Automatic Fixes Built In

| Issue | Fix Applied Automatically |
|-------|--------------------------|
| WebM/MP4/MOV variable bitrate | Converted to MP3 via ffmpeg before transcription |
| CPU FP16 warning / slow inference | Detects CPU vs GPU, uses int8 on CPU and float16 on GPU |
| Wrong language detection | Defaults to English, skips auto-detection |

---

## Cost

Transcription (faster-whisper) is always **free**. The DeepSeek V3 API call is the only cost.

| Recording Length | Estimated Cost |
|-----------------|----------------|
| 30 minutes | ~$0.003 |
| 1 hour | ~$0.007 |
| 2 hours | ~$0.012 |

---

## Output

Reports are saved as markdown files:

- **Media file input:** saved next to the source file as `<filename>_analysis_<timestamp>.md`
- **Transcript input (UI):** saved to your Downloads folder as `transcript_analysis_<timestamp>.md`

---

## Supported Formats

Any format supported by ffmpeg: `mp3`, `mp4`, `webm`, `wav`, `m4a`, `mov`, `mkv`, `flac`, `ogg`, and more.
