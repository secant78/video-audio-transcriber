# Video/Audio Transcript Analyzer

Transcribes recorded audio or video files and generates a comprehensive analysis report using OpenAI Whisper (local, free) and the Claude API.

## What It Does

Given any audio or video file, the program produces a structured markdown report with:

1. **Full Transcript** - timestamped transcription of the entire recording
2. **Detailed Summary** - 8-12 sentence narrative overview of the session
3. **Questions & Answers** - every Q&A pair extracted with timestamps
4. **Tools, Technologies & Use Cases** - every tool/tech mentioned, its category, how it was used, and general capabilities
5. **Technical Deep Dive** - in-depth discussion of every major technical topic covered
6. **What Was Done Well** - specific strengths with references to the transcript
7. **Areas for Improvement** - constructive, actionable feedback with suggestions
8. **Overall Assessment** - final evaluation and recommended priority action

## Prerequisites

**Python 3.8+** and **ffmpeg** must be installed.

Install ffmpeg (if not already installed):
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

You will also need an **Anthropic API key** from [console.anthropic.com](https://console.anthropic.com).

## Usage

```bash
# Set your API key (do this once per terminal session)
set ANTHROPIC_API_KEY=sk-ant-...        # Windows
export ANTHROPIC_API_KEY=sk-ant-...     # Mac/Linux

# Run a full analysis
python analyzer.py "recording.mp3"

# Specify a video file
python analyzer.py "lecture.mp4"

# Use a more accurate Whisper model (recommended for technical content)
python analyzer.py "recording.mp3" --model medium

# Skip Claude and only produce the transcript (no API key needed)
python analyzer.py "recording.mp3" --transcript-only

# Pass the API key inline instead of via env var
python analyzer.py "recording.mp3" --api-key sk-ant-...
```

## Whisper Model Sizes

Whisper runs locally and is completely free. Larger models are slower but more accurate.

| Model | Download Size | Speed | Best For |
|-------|-------------|-------|----------|
| `tiny` | 39 MB | Fastest | Quick drafts |
| `base` | 74 MB | Fast | Default, good balance |
| `small` | 244 MB | Moderate | Better accuracy |
| `medium` | 769 MB | Slow | Technical/dense audio (recommended) |
| `large` | 1.5 GB | Slowest | Maximum accuracy |

Models are downloaded automatically on first use and cached locally.

## Cost

Transcription (Whisper) is always **free**. The Claude API call is the only cost.

| Recording Length | Estimated Cost |
|-----------------|----------------|
| 30 minutes | ~$0.05 |
| 1 hour | ~$0.08 |
| 2 hours | ~$0.13 |

## Output

The report is saved as a markdown file in the same directory as the input file:

```
recording_analysis_20260505_143022.md
```

## Supported Formats

Any format supported by ffmpeg: `mp3`, `mp4`, `wav`, `m4a`, `mov`, `mkv`, `webm`, `flac`, `ogg`, and more.
