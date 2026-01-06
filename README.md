# Audio Transcriber

從 YouTube 或 Podcast 下載音頻，並使用本地 Whisper 模型進行語音轉錄，提取字幕。

## 安裝

### 1. 安裝 FFmpeg（必需）

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

### 2. 安裝 Deno（YouTube 下載需要）

YouTube 使用 JavaScript 挑戰來保護影片，yt-dlp 需要 Deno 來解決這些挑戰。

```bash
# macOS
brew install deno

# 其他系統
curl -fsSL https://deno.land/install.sh | sh
```

### 3. 安裝 Python 依賴

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本使用

```bash
# 轉錄 YouTube 影片
python -m audio_transcriber "https://www.youtube.com/watch?v=VIDEO_ID"

# 轉錄本地音頻檔案
python -m audio_transcriber audio.mp3
```

### 進階選項

```bash
# 使用較大的模型（更準確）
python -m audio_transcriber "URL" --model large-v3

# 翻譯成英文
python -m audio_transcriber "URL" --translate

# 指定源語言（跳過自動檢測）
python -m audio_transcriber "URL" --language ja

# 輸出為 SRT 字幕格式
python -m audio_transcriber "URL" --format srt

# 指定輸出目錄
python -m audio_transcriber "URL" --output ./subtitles
```

### 可用模型

| 模型     | 大小  | 記憶體需求 | 相對速度   |
| -------- | ----- | ---------- | ---------- |
| tiny     | 39M   | ~1GB       | 最快       |
| base     | 74M   | ~1GB       | 快         |
| small    | 244M  | ~2GB       | 中等       |
| medium   | 769M  | ~5GB       | 慢         |
| large-v3 | 1550M | ~10GB      | 最慢但最準 |

## 輸出格式

- **txt**: 純文字（預設）
- **srt**: SubRip 字幕格式
- **vtt**: WebVTT 字幕格式
- **json**: 包含時間戳的 JSON 格式

## 範例

```bash
# 轉錄日文 YouTube 影片並翻譯成英文
python -m audio_transcriber "https://www.youtube.com/watch?v=xxx" \
    --language ja \
    --translate \
    --format srt \
    --model medium
```
