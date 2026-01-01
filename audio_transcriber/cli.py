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
    srt_to_txt,
    USE_MLX,
    is_apple_silicon,
    has_mlx_whisper,
    MLX_MODEL_MAP,
)

console = Console()


@click.group()
def cli():
    """Audio Transcriber - Download and transcribe audio with local Whisper."""
    pass


@click.command()
@click.argument('source')
@click.option(
    '--model', '-m',
    default='large-v3-turbo',
    type=click.Choice(MODEL_SIZES),
    help='Whisper model size (default: large-v3-turbo)'
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
@click.option(
    '--beam-size', '-b',
    default=5,
    type=int,
    help='Beam size for decoding (1=fastest, 5=default)'
)
@click.option(
    '--fast',
    is_flag=True,
    help='Fast mode: use large-v3-turbo model with beam_size=1'
)
def aits(
    source: str,
    model: str,
    language: Optional[str],
    translate: bool,
    output: Optional[str],
    keep_audio: bool,
    dry_run: bool,
    beam_size: int,
    fast: bool,
):
    """
    Download and transcribe audio from YouTube, Podcast, or local files.

    SOURCE can be a URL (YouTube, etc.) or a local audio file path.

    Examples:

        aits "https://youtube.com/watch?v=..."

        aits "URL" --translate

        aits "URL" --model large-v3
    """
    # Detect platform and backend
    is_mac = is_apple_silicon()
    mlx_available = has_mlx_whisper()
    backend = "mlx-whisper" if USE_MLX else "faster-whisper"

    # Show model info
    if USE_MLX:
        model_display = MLX_MODEL_MAP.get(model, model)
    else:
        model_display = model

    console.print(Panel.fit(
        "[bold cyan]🎙️ Audio Transcriber[/]\n"
        "[dim]Download & transcribe audio with local Whisper[/]\n\n"
        f"[dim]Platform:[/] {'Apple Silicon' if is_mac else 'Other'} | "
        f"[dim]MLX:[/] {'✓' if mlx_available else '✗'} | "
        f"[dim]Backend:[/] [bold]{backend}[/]\n"
        f"[dim]Model:[/] [bold]{model_display}[/]",
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
                duration = int(info['duration'])
                console.print(f"  Duration: {duration // 60}:{duration % 60:02d}")
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

        # Fast mode overrides
        if fast:
            model = 'large-v3-turbo'
            beam_size = 1
            console.print("[yellow]⚡ Fast mode: using large-v3-turbo with beam_size=1[/]")

        result = transcribe(
            audio_path,
            model_size=model,
            language=language,
            translate=translate,
            beam_size=beam_size,
        )
        
        # Save transcription
        console.print("\n[bold]Step 3: Saving transcription[/]")

        # Generate output filename
        base_name = Path(audio_path).stem
        output_path = os.path.join(output, base_name)

        # Always save SRT first
        saved_srt = save_transcription(result, output_path, 'srt')

        # Then convert to TXT
        txt_path = str(Path(output_path).with_suffix('.txt'))
        with open(saved_srt, 'r', encoding='utf-8') as f:
            srt_content = f.read()
        txt_content = srt_to_txt(srt_content)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(txt_content)
        console.print(f"[bold green]💾 Saved:[/] {txt_path}")

        # Show summary
        console.print("\n" + "─" * 50)
        elapsed = result.get('elapsed_time', 0)
        speed = result.get('speed_ratio', 0)
        console.print(Panel(
            f"[bold green]✅ Transcription complete![/]\n\n"
            f"📁 SRT: [cyan]{saved_srt}[/]\n"
            f"📁 TXT: [cyan]{txt_path}[/]\n"
            f"🌐 Language: {result['language']}\n"
            f"⏱️  Audio duration: {result['duration']:.1f}s\n"
            f"🚀 Processing time: {elapsed:.1f}s ({speed:.1f}x realtime)\n"
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


@cli.command('strip-srt')
@click.argument('srt_file')
@click.option(
    '--output', '-o',
    default=None,
    help='Output file path (default: same name with .txt extension)'
)
def strip_srt(srt_file: str, output: Optional[str]):
    """
    Convert SRT file to plain text, removing timestamps and sequence numbers.

    Example:
        python -m audio_transcriber strip-srt subtitle.srt
    """
    if not os.path.exists(srt_file):
        console.print(f"[bold red]❌ File not found:[/] {srt_file}")
        sys.exit(1)

    with open(srt_file, 'r', encoding='utf-8') as f:
        srt_content = f.read()

    txt_content = srt_to_txt(srt_content)

    if output is None:
        output = str(Path(srt_file).with_suffix('.txt'))

    with open(output, 'w', encoding='utf-8') as f:
        f.write(txt_content)

    console.print(f"[bold green]✅ Converted:[/] {output}")


if __name__ == '__main__':
    cli()
