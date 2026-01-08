# Audio Transcriber

One command to turn any video into text.

```bash
aits "https://www.youtube.com/watch?v=VIDEO_ID"
```

## Why This Tool?

| Feature          | Description                                                       |
| ---------------- | ----------------------------------------------------------------- |
| **Fast**         | ~30s to transcribe 10 min audio on Apple Silicon (20x realtime)   |
| **Accurate**     | OpenAI Whisper large-v3-turbo, supports 99 languages              |
| **Wide Support** | YouTube, Bilibili, Twitter, TikTok, and 1000+ platforms           |
| **Free**         | Runs entirely local, no API costs, no usage limits                |
| **Private**      | Audio never leaves your machine, fully offline                    |

## Supported Sources

**Online Platforms** (via yt-dlp):
- Video: YouTube, Bilibili, Vimeo, Dailymotion, TikTok
- Social: Twitter/X, Instagram, Facebook, Reddit
- Live: Twitch, Kick (VODs)
- Music: SoundCloud, Bandcamp
- Podcast: Apple Podcasts,  RSS Feeds, mp3 files

**Local Files**: mp3, wav, m4a, mp4, mkv, and more

Full list: [yt-dlp supported sites](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)

## Speed Reference

On Apple Silicon Mac (M1/M2/M3) with default model:

| Audio Length | Processing Time | Speed |
| ------------ | --------------- | ----- |
| 10 min       | ~0.5 min        | 20x   |
| 60 min       | ~3 min          | 20x   |
| 120 min      | ~6 min          | 20x   |

Use `--fast` mode for even faster processing with slightly lower accuracy.

## Installation

```bash
pip install audio-transcriber
```

That's it. The tool auto-installs dependencies (ffmpeg, deno, mlx-whisper) as needed.

## Usage

```bash
# Simple: just paste a URL
aits "https://www.youtube.com/watch?v=VIDEO_ID"

# Transcribe local file
aits audio.mp3

# Fast mode (trade accuracy for speed)
aits "URL" --fast

# Japanese video translated to English
aits "URL" --language ja --translate

# Output SRT subtitles (for video editors)
aits "URL" --srt
```

### All Options

| Option         | Short | Description                                  |
| -------------- | ----- | -------------------------------------------- |
| `--model`      | `-m`  | Model size (default: large-v3-turbo)         |
| `--language`   | `-l`  | Source language (e.g., en, ja, zh)           |
| `--translate`  | `-t`  | Translate to English                         |
| `--output`     | `-o`  | Output directory (default: ./output)         |
| `--fast`       |       | Fast mode (turbo + beam_size=1)              |
| `--srt`        |       | Also output SRT subtitle file                |
| `--keep-audio` |       | Keep downloaded audio file                   |
| `--dry-run`    |       | Show info only, don't download or transcribe |

## Model Selection

| Model             | Speed | Accuracy | Recommended For                         |
| ----------------- | ----- | -------- | --------------------------------------- |
| `large-v3-turbo`  | ★★★★  | ★★★★★    | **Default**, best speed/quality balance |
| `distil-large-v3` | ★★★★  | ★★★★     | Fast high-quality alternative           |
| `medium`          | ★★★   | ★★★★     | Lower memory usage                      |
| `small`           | ★★★★  | ★★★      | Quick drafts                            |
| `base`            | ★★★★★ | ★★       | Very fast, lower quality                |
| `tiny`            | ★★★★★ | ★        | Testing only                            |

First run downloads the model (~1.5GB), then uses cache.

## System Requirements

- Python 3.9+
- FFmpeg (audio processing)
- Deno (required for YouTube downloads)

### Platform Support

| Platform          | Backend        | Notes                    |
| ----------------- | -------------- | ------------------------ |
| Apple Silicon Mac | mlx-whisper    | GPU accelerated, fastest |
| Intel Mac         | faster-whisper | CPU                      |
| Linux             | faster-whisper | CUDA GPU supported       |
| Windows           | faster-whisper | CUDA GPU supported       |

## Output Formats

- **txt**: Plain text (default)
- **srt**: SubRip subtitles, works with Premiere, Final Cut, DaVinci Resolve, etc.

## Other Tools

### SRT to Plain Text

```bash
python -m audio_transcriber strip-srt subtitle.srt
```

---

[中文版 README](README.zh-TW.md)
