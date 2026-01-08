# Audio Transcriber

一行指令，把任何影片變成文字。

```bash
aits "https://www.youtube.com/watch?v=VIDEO_ID"
```

## 為什麼選擇這個工具？

| 特點     | 說明                                                       |
| -------- | ---------------------------------------------------------- |
| **快**   | Apple Silicon 上 10 分鐘音訊約 30 秒轉完（20x 即時速度） |
| **準**   | 使用 OpenAI Whisper large-v3-turbo，支援 99 種語言        |
| **廣**   | 支援 YouTube、Bilibili、Twitter、TikTok 等 1000+ 平台     |
| **免費** | 完全本地運行，無 API 費用，無使用限制                     |
| **隱私** | 音訊不上傳任何伺服器，全程離線處理                        |

## 支援的來源

**網路平台**（透過 yt-dlp）：
- 影片：YouTube、Bilibili、Vimeo、Dailymotion、TikTok
- 社群：Twitter/X、Instagram、Facebook、Reddit
- 直播：Twitch、Kick（VOD 回放）
- 音樂：SoundCloud、Bandcamp
- Podcast：Apple Podcasts、RSS Feed、mp3 結尾的網址

**本地檔案**：mp3、wav、m4a、mp4、mkv 等常見格式

完整支援清單見 [yt-dlp 支援網站](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)

## 轉錄速度參考

在 Apple Silicon Mac（M1/M2/M3）上使用預設模型：

| 音訊長度  | 處理時間  | 速度 |
| --------- | --------- | ---- |
| 10 分鐘   | ~0.5 分鐘 | 20x  |
| 60 分鐘   | ~3 分鐘   | 20x  |
| 120 分鐘  | ~6 分鐘   | 20x  |

使用 `--fast` 模式可更快，但準確度略降。

## 安裝

```bash
pip install audio-transcriber
```

就這樣。程式會自動安裝缺少的依賴（ffmpeg、deno、mlx-whisper）。

## 使用方法

```bash
# 最簡單：貼上網址就好
aits "https://www.youtube.com/watch?v=VIDEO_ID"

# 轉錄本地檔案
aits audio.mp3

# 快速模式（犧牲一點準確度換速度）
aits "URL" --fast

# 日文影片翻譯成英文
aits "URL" --language ja --translate

# 輸出 SRT 字幕檔（可匯入剪輯軟體）
aits "URL" --srt
```

### 所有選項

| 選項           | 簡寫 | 說明                             |
| -------------- | ---- | -------------------------------- |
| `--model`      | `-m` | 模型大小（預設：large-v3-turbo） |
| `--language`   | `-l` | 指定來源語言（如 en、ja、zh）    |
| `--translate`  | `-t` | 翻譯成英文                       |
| `--output`     | `-o` | 輸出目錄（預設：./output）       |
| `--fast`       |      | 快速模式（turbo + beam_size=1）  |
| `--srt`        |      | 同時輸出 SRT 字幕檔              |
| `--keep-audio` |      | 保留下載的音訊檔                 |
| `--dry-run`    |      | 只顯示資訊，不下載或轉錄         |

## 模型選擇

| 模型              | 速度  | 準確度 | 建議用途                         |
| ----------------- | ----- | ------ | -------------------------------- |
| `large-v3-turbo`  | ★★★★  | ★★★★★  | **預設推薦**，速度與品質最佳平衡 |
| `distil-large-v3` | ★★★★  | ★★★★   | 高速高品質替代方案               |
| `medium`          | ★★★   | ★★★★   | 記憶體較少時使用                 |
| `small`           | ★★★★  | ★★★    | 快速草稿                         |
| `base`            | ★★★★★ | ★★     | 超快但品質一般                   |
| `tiny`            | ★★★★★ | ★      | 僅供測試                         |

首次使用模型會自動下載（約 1.5GB），之後直接使用快取。

## 系統需求

- Python 3.9+
- FFmpeg（音訊處理）
- Deno（YouTube 下載需要）

### 平台支援

| 平台              | 後端           | 備註               |
| ----------------- | -------------- | ------------------ |
| Apple Silicon Mac | mlx-whisper    | GPU 加速，速度最快 |
| Intel Mac         | faster-whisper | CPU 運算           |
| Linux             | faster-whisper | 支援 CUDA GPU 加速 |
| Windows           | faster-whisper | 支援 CUDA GPU 加速 |

## 輸出格式

- **txt**：純文字（預設）
- **srt**：SubRip 字幕格式，可匯入 Premiere、Final Cut、剪映等

## 其他工具

### SRT 轉純文字

```bash
python -m audio_transcriber strip-srt subtitle.srt
```

---

[English README](README.md)
