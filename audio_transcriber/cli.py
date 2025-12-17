"""CLI interface for audio transcriber."""

import os
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel

from .downloader import download_audio, is_url, get_audio_info
from .transcriber import (
    transcribe,
    save_transcription,
    load_model,
    MODEL_SIZES,
)

console = Console()


@click.command()
@click.argument('source')
@click.option(
    '--model', '-m',
    default='medium',
    type=click.Choice(MODEL_SIZES),
    help='Whisper model size (default: medium)'
)
@click.option(
    '--language', '-l',
    default=None,
    help='Source language code (e.g., en, ja, zh). Auto-detect if not specified.'
)
@click.option(
    '--translate', '-t',
    is_flag=True,
    help='Translate to English'
)
@click.option(
    '--format', '-f',
    default='srt',
    type=click.Choice(['srt', 'vtt', 'txt', 'json']),
    help='Output format (default: srt)'
)
@click.option(
    '--output', '-o',
    default=None,
    help='Output directory (default: ./output)'
)
@click.option(
    '--keep-audio',
    is_flag=True,
    help='Keep downloaded audio file'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Show info without downloading or transcribing'
)
def main(
    source: str,
    model: str,
    language: Optional[str],
    translate: bool,
    format: str,
    output: Optional[str],
    keep_audio: bool,
    dry_run: bool
):
    """
    Download and transcribe audio from YouTube, Podcast, or local files.
    
    SOURCE can be a URL (YouTube, etc.) or a local audio file path.
    
    Examples:
    
        # Transcribe a YouTube video
        python -m audio_transcriber "https://youtube.com/watch?v=..."
        
        # Transcribe with translation to English
        python -m audio_transcriber "URL" --translate
        
        # Use a specific model and output format
        python -m audio_transcriber "URL" --model large-v3 --format vtt
    """
    console.print(Panel.fit(
        "[bold cyan]🎙️ Audio Transcriber[/]\n"
        "[dim]Download & transcribe audio with local Whisper[/]",
        border_style="cyan"
    ))
    
    # Set default output directory
    if output is None:
        output = os.path.join(os.getcwd(), 'output')
    os.makedirs(output, exist_ok=True)
    
    audio_path = None
    downloaded = False
    
    try:
        # Check if source is URL or local file
        if is_url(source):
            if dry_run:
                console.print("\n[bold]📋 URL Info:[/]")
                info = get_audio_info(source)
                console.print(f"  Title: {info['title']}")
                console.print(f"  Duration: {info['duration'] // 60}:{info['duration'] % 60:02d}")
                console.print(f"  Uploader: {info['uploader']}")
                return
            
            # Download audio
            console.print("\n[bold]Step 1: Downloading audio[/]")
            audio_path = download_audio(source, output)
            downloaded = True
        else:
            # Local file
            if not os.path.exists(source):
                console.print(f"[bold red]❌ File not found:[/] {source}")
                sys.exit(1)
            audio_path = source
            
            if dry_run:
                console.print(f"\n[bold]📋 Local file:[/] {source}")
                return
        
        # Transcribe
        console.print("\n[bold]Step 2: Transcribing audio[/]")
        
        result = transcribe(
            audio_path,
            model_size=model,
            language=language,
            translate=translate
        )
        
        # Save transcription
        console.print("\n[bold]Step 3: Saving transcription[/]")
        
        # Generate output filename
        base_name = Path(audio_path).stem
        output_path = os.path.join(output, base_name)
        
        saved_path = save_transcription(result, output_path, format)
        
        # Show summary
        console.print("\n" + "─" * 50)
        console.print(Panel(
            f"[bold green]✅ Transcription complete![/]\n\n"
            f"📁 Output: [cyan]{saved_path}[/]\n"
            f"🌐 Language: {result['language']}\n"
            f"⏱️  Duration: {result['duration']:.1f}s\n"
            f"📝 Segments: {len(result['segments'])}",
            title="Summary",
            border_style="green"
        ))
        
        # Preview first few lines
        console.print("\n[bold]Preview:[/]")
        for seg in result['segments'][:3]:
            console.print(f"  [{seg['start']:.1f}s] {seg['text']}")
        if len(result['segments']) > 3:
            console.print(f"  ... and {len(result['segments']) - 3} more segments")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Cancelled by user[/]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/] {e}")
        sys.exit(1)
    finally:
        # Clean up downloaded audio if not keeping
        if downloaded and not keep_audio and audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
            console.print(f"[dim]🗑️  Cleaned up temporary audio file[/]")


if __name__ == '__main__':
    main()
