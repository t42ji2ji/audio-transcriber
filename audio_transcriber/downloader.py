"""Audio downloader module using yt-dlp."""

import os
import re
import tempfile
from pathlib import Path
from typing import Optional

import yt_dlp
from rich.console import Console

console = Console()


def is_url(path: str) -> bool:
    """Check if the given path is a URL."""
    return path.startswith(('http://', 'https://', 'www.'))


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing invalid characters."""
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    return filename


def download_audio(
    url: str,
    output_dir: Optional[str] = None,
    keep_original: bool = False
) -> str:
    """
    Download audio from a URL (YouTube, Podcast, etc.).
    
    Args:
        url: The URL to download from
        output_dir: Directory to save the audio file (default: ./output)
        keep_original: Keep the original format instead of converting to WAV
    
    Returns:
        Path to the downloaded audio file
    """
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'output')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # yt-dlp options for audio extraction
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'quiet': False,
        'no_warnings': False,
        'extract_flat': False,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav' if not keep_original else 'mp3',
            'preferredquality': '192',
        }],
        # Convert to 16kHz mono for Whisper
        'postprocessor_args': [
            '-ar', '16000',
            '-ac', '1',
        ] if not keep_original else [],
    }
    
    console.print(f"[bold blue]⬇️  Downloading audio from:[/] {url}")
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info first to get the title
            info = ydl.extract_info(url, download=False)
            title = sanitize_filename(info.get('title', 'audio'))
            
            console.print(f"[dim]Title: {info.get('title', 'Unknown')}[/]")
            console.print(f"[dim]Duration: {info.get('duration', 0) // 60}:{info.get('duration', 0) % 60:02d}[/]")
            
            # Download
            ydl.download([url])
            
            # Find the downloaded file
            ext = 'wav' if not keep_original else 'mp3'
            output_path = os.path.join(output_dir, f"{title}.{ext}")
            
            # Handle case where filename might differ
            if not os.path.exists(output_path):
                # Try to find any audio file in the output directory
                for file in os.listdir(output_dir):
                    if file.endswith(f'.{ext}'):
                        output_path = os.path.join(output_dir, file)
                        break
            
            console.print(f"[bold green]✅ Downloaded:[/] {output_path}")
            return output_path
            
    except yt_dlp.DownloadError as e:
        console.print(f"[bold red]❌ Download failed:[/] {e}")
        raise
    except Exception as e:
        console.print(f"[bold red]❌ Error:[/] {e}")
        raise


def get_audio_info(url: str) -> dict:
    """
    Get information about an audio/video without downloading.
    
    Args:
        url: The URL to get info from
    
    Returns:
        Dictionary with title, duration, etc.
    """
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            'title': info.get('title'),
            'duration': info.get('duration'),
            'uploader': info.get('uploader'),
            'description': info.get('description', '')[:200],
        }


if __name__ == '__main__':
    # Test with a short video
    import sys
    if len(sys.argv) > 1:
        download_audio(sys.argv[1])
    else:
        print("Usage: python downloader.py <URL>")
