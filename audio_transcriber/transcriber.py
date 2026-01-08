"""Transcription module using faster-whisper or mlx-whisper."""

import os
import platform
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

console = Console()


def _get_hf_cache_dir() -> Path:
    """Get the HuggingFace cache directory."""
    # Check environment variables first
    if os.environ.get('HF_HUB_CACHE'):
        return Path(os.environ['HF_HUB_CACHE'])
    if os.environ.get('HF_HOME'):
        return Path(os.environ['HF_HOME']) / 'hub'
    # Default location
    return Path.home() / '.cache' / 'huggingface' / 'hub'


def _is_model_cached(repo_id: str) -> bool:
    """Check if a HuggingFace model is already cached locally."""
    cache_dir = _get_hf_cache_dir()
    # HuggingFace cache uses format: models--{org}--{repo}
    model_cache_name = f"models--{repo_id.replace('/', '--')}"
    model_cache_path = cache_dir / model_cache_name

    # Check if the directory exists and has content
    if model_cache_path.exists():
        # Check for snapshots directory which indicates a completed download
        snapshots_dir = model_cache_path / 'snapshots'
        if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
            return True
    return False


def _get_model_size_mb(model_size: str) -> int:
    """Get approximate model size in MB."""
    sizes = {
        'tiny': 75,
        'base': 140,
        'small': 460,
        'medium': 1500,
        'large-v3-turbo': 1500,
        'distil-large-v3': 1500,
    }
    return sizes.get(model_size, 1500)


def _show_download_notice(model_size: str, repo_id: str):
    """Show a notice that the model will be downloaded."""
    size_mb = _get_model_size_mb(model_size)
    size_str = f"{size_mb} MB" if size_mb < 1000 else f"{size_mb / 1000:.1f} GB"

    console.print()
    console.print(Panel(
        f"[bold yellow]⬇️  First-time model download[/]\n\n"
        f"Model [cyan]{repo_id}[/] is not cached locally.\n"
        f"Downloading [bold]{size_str}[/] (one-time only).\n\n"
        f"[dim]The model will be cached at:\n"
        f"~/.cache/huggingface/hub/[/]",
        title="📦 Model Download",
        border_style="yellow"
    ))
    console.print()

# Detect if running on Apple Silicon
def is_apple_silicon() -> bool:
    return platform.system() == 'Darwin' and platform.machine() == 'arm64'


# Check if mlx-whisper is available (re-check each time, in case it was just installed)
def has_mlx_whisper() -> bool:
    try:
        import mlx_whisper
        return True
    except ImportError:
        return False


def should_use_mlx() -> bool:
    """Determine if MLX backend should be used (checked dynamically)."""
    return is_apple_silicon() and has_mlx_whisper()


# For backwards compatibility - but prefer should_use_mlx() for dynamic check
USE_MLX = should_use_mlx()

# Available model sizes (ordered by speed, fastest first)
MODEL_SIZES = [
    'tiny',
    'base',
    'small',
    'medium',
    'large-v3-turbo',  # 6x faster than large-v3, recommended
    'distil-large-v3', # 6x faster than large-v3, <1% WER difference
]

# MLX model mapping (model_size -> HuggingFace repo)
MLX_MODEL_MAP = {
    'tiny': 'mlx-community/whisper-tiny',
    'base': 'mlx-community/whisper-base',
    'small': 'mlx-community/whisper-small',
    'medium': 'mlx-community/whisper-medium',
    'large-v3-turbo': 'mlx-community/whisper-large-v3-turbo',
    'distil-large-v3': 'mlx-community/distil-whisper-large-v3',
}

# faster-whisper model mapping (model_size -> HuggingFace repo)
FASTER_WHISPER_MODEL_MAP = {
    'tiny': 'Systran/faster-whisper-tiny',
    'base': 'Systran/faster-whisper-base',
    'small': 'Systran/faster-whisper-small',
    'medium': 'Systran/faster-whisper-medium',
    'large-v3-turbo': 'mobiuslabsgmbh/faster-whisper-large-v3-turbo',
    'distil-large-v3': 'Systran/faster-distil-whisper-large-v3',
}


def load_model(
    model_size: str = 'large-v3-turbo',
    device: str = 'auto',
    compute_type: str = 'auto'
) -> Union['WhisperModel', None]:
    """
    Load the Whisper model (faster-whisper only, mlx-whisper doesn't need preloading).

    Args:
        model_size: Model size (tiny, base, small, medium, large-v3-turbo, distil-large-v3)
        device: Device to use (auto, cpu, cuda) - ignored for mlx
        compute_type: Compute type (auto, int8, float16, float32) - ignored for mlx

    Returns:
        Loaded WhisperModel (or None for mlx-whisper)
    """
    if model_size not in MODEL_SIZES:
        console.print(f"[yellow]⚠️  Unknown model size '{model_size}', using 'large-v3-turbo'[/]")
        model_size = 'large-v3-turbo'

    # Check dynamically (in case mlx was just installed)
    use_mlx = should_use_mlx()

    if use_mlx:
        # mlx-whisper loads model during transcribe, no preloading needed
        console.print(f"[bold blue]🔄 Using MLX backend:[/] {model_size}")
        console.print(f"[dim]Backend: mlx-whisper (Apple Silicon optimized)[/]")
        return None

    console.print(f"[bold blue]🔄 Loading model:[/] {model_size}")

    # Auto-detect best settings
    if device == 'auto':
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            device = 'cpu'

    if compute_type == 'auto':
        compute_type = 'int8' if device == 'cpu' else 'float16'

    console.print(f"[dim]Backend: faster-whisper, Device: {device}, Compute type: {compute_type}[/]")

    # Check if model needs to be downloaded
    repo_id = FASTER_WHISPER_MODEL_MAP.get(model_size, f'Systran/faster-whisper-{model_size}')
    if not _is_model_cached(repo_id):
        _show_download_notice(model_size, repo_id)

    # Import here to allow dynamic backend selection
    from faster_whisper import WhisperModel

    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type
    )

    console.print(f"[bold green]✅ Model loaded[/]")
    return model


def transcribe(
    audio_path: str,
    model: Optional[Any] = None,
    model_size: str = 'large-v3-turbo',
    language: Optional[str] = None,
    translate: bool = False,
    word_timestamps: bool = False,
    beam_size: int = 5,
) -> Dict[str, Any]:
    """
    Transcribe an audio file.

    Args:
        audio_path: Path to the audio file
        model: Pre-loaded WhisperModel (optional, ignored for mlx-whisper)
        model_size: Model size if model not provided
        language: Source language (None for auto-detection)
        translate: If True, translate to English
        word_timestamps: If True, include word-level timestamps
        beam_size: Beam size for decoding (1=fastest, 5=default, higher=more accurate)

    Returns:
        Dictionary with transcription results:
        {
            'text': full text,
            'segments': list of segments with timestamps,
            'language': detected language
        }
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if model_size not in MODEL_SIZES:
        console.print(f"[yellow]⚠️  Unknown model size '{model_size}', using 'large-v3-turbo'[/]")
        model_size = 'large-v3-turbo'

    console.print(f"[bold blue]🎤 Transcribing:[/] {audio_path}")

    # Check dynamically (in case mlx was just installed)
    if should_use_mlx():
        return _transcribe_mlx(
            audio_path=audio_path,
            model_size=model_size,
            language=language,
            translate=translate,
            word_timestamps=word_timestamps,
        )
    else:
        return _transcribe_faster_whisper(
            audio_path=audio_path,
            model=model,
            model_size=model_size,
            language=language,
            translate=translate,
            word_timestamps=word_timestamps,
            beam_size=beam_size,
        )


def _transcribe_mlx(
    audio_path: str,
    model_size: str,
    language: Optional[str],
    translate: bool,
    word_timestamps: bool,
) -> Dict[str, Any]:
    """Transcribe using mlx-whisper backend."""
    import mlx_whisper

    mlx_model = MLX_MODEL_MAP.get(model_size, MLX_MODEL_MAP['large-v3-turbo'])
    console.print(f"[dim]Backend: mlx-whisper, Model: {mlx_model}[/]")

    # Check if model needs to be downloaded
    if not _is_model_cached(mlx_model):
        _show_download_notice(model_size, mlx_model)

    task = 'translate' if translate else 'transcribe'

    # Get audio duration first for progress estimation
    import mlx_whisper.audio as mlx_audio
    audio_array = mlx_audio.load_audio(audio_path)
    total_duration = len(audio_array) / 16000  # 16kHz sample rate

    # Estimate processing time based on typical 20x realtime speed for MLX
    estimated_time = total_duration / 20

    start_time = time.time()

    import threading
    from rich.live import Live
    from rich.text import Text

    result_holder = [None]
    error_holder = [None]

    def do_transcribe():
        try:
            result_holder[0] = mlx_whisper.transcribe(
                audio_path,
                path_or_hf_repo=mlx_model,
                language=language,
                task=task,
                word_timestamps=word_timestamps,
            )
        except Exception as e:
            error_holder[0] = e

    transcribe_thread = threading.Thread(target=do_transcribe)
    transcribe_thread.start()

    with Live(console=console, refresh_per_second=4, transient=True) as live:
        while transcribe_thread.is_alive():
            elapsed = time.time() - start_time
            # Estimate progress based on elapsed time vs estimated time
            progress_pct = min(99, (elapsed / estimated_time) * 100) if estimated_time > 0 else 0
            progress_bar = "█" * int(progress_pct / 5) + "░" * (20 - int(progress_pct / 5))
            live.update(Text(
                f"⏳ Transcribing {total_duration:.0f}s audio... [{progress_bar}] ~{progress_pct:.0f}% ({elapsed:.1f}s)",
                style="bold blue"
            ))
            time.sleep(0.25)

    transcribe_thread.join()

    if error_holder[0]:
        raise error_holder[0]

    result = result_holder[0]
    elapsed_time = time.time() - start_time

    # Normalize result format
    segments = result.get('segments', [])
    segment_list = []
    for i, seg in enumerate(segments):
        segment_dict = {
            'id': i,
            'start': seg['start'],
            'end': seg['end'],
            'text': seg['text'].strip(),
        }
        if word_timestamps and 'words' in seg:
            segment_dict['words'] = [
                {'word': w['word'], 'start': w['start'], 'end': w['end']}
                for w in seg['words']
            ]
        segment_list.append(segment_dict)

    # Calculate duration from last segment
    duration = segment_list[-1]['end'] if segment_list else 0.0
    detected_language = result.get('language', 'unknown')

    # Calculate speed ratio (audio duration / processing time)
    speed_ratio = duration / elapsed_time if elapsed_time > 0 else 0

    output = {
        'text': result.get('text', ''),
        'segments': segment_list,
        'language': detected_language,
        'language_probability': 1.0,  # mlx-whisper doesn't provide this
        'duration': duration,
        'elapsed_time': elapsed_time,
        'speed_ratio': speed_ratio,
    }

    console.print(f"[bold green]✅ Transcription complete[/]")
    console.print(f"[dim]Detected language: {detected_language}[/]")
    console.print(f"[dim]Audio duration: {duration:.1f}s, Segments: {len(segment_list)}[/]")
    console.print(f"[dim]Processing time: {elapsed_time:.1f}s ({speed_ratio:.1f}x realtime)[/]")

    return output


def _transcribe_faster_whisper(
    audio_path: str,
    model: Optional[Any],
    model_size: str,
    language: Optional[str],
    translate: bool,
    word_timestamps: bool,
    beam_size: int,
) -> Dict[str, Any]:
    """Transcribe using faster-whisper backend."""
    # Load model if not provided
    if model is None:
        model = load_model(model_size)

    task = 'translate' if translate else 'transcribe'

    start_time = time.time()

    # First, get audio duration for progress tracking
    segments, info = model.transcribe(
        audio_path,
        language=language,
        task=task,
        word_timestamps=word_timestamps,
        beam_size=beam_size,
        vad_filter=True,  # Filter out silence
    )

    duration = info.duration

    # Convert generator to list with progress bar
    segment_list = []
    full_text = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TextColumn("[cyan]{task.completed:.1f}s[/] / [cyan]{task.total:.1f}s[/]"),
        console=console,
        transient=False
    ) as progress:
        task_id = progress.add_task(
            description="Transcribing...",
            total=duration
        )

        for segment in segments:
            segment_dict = {
                'id': segment.id,
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip(),
            }

            if word_timestamps and segment.words:
                segment_dict['words'] = [
                    {'word': w.word, 'start': w.start, 'end': w.end}
                    for w in segment.words
                ]

            segment_list.append(segment_dict)
            full_text.append(segment.text.strip())

            # Update progress based on segment end time
            progress.update(task_id, completed=segment.end)

    elapsed_time = time.time() - start_time
    speed_ratio = duration / elapsed_time if elapsed_time > 0 else 0

    result = {
        'text': ' '.join(full_text),
        'segments': segment_list,
        'language': info.language,
        'language_probability': info.language_probability,
        'duration': info.duration,
        'elapsed_time': elapsed_time,
        'speed_ratio': speed_ratio,
    }

    console.print(f"[bold green]✅ Transcription complete[/]")
    console.print(f"[dim]Detected language: {info.language} ({info.language_probability:.1%})[/]")
    console.print(f"[dim]Audio duration: {info.duration:.1f}s, Segments: {len(segment_list)}[/]")
    console.print(f"[dim]Processing time: {elapsed_time:.1f}s ({speed_ratio:.1f}x realtime)[/]")

    return result


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Convert seconds to VTT timestamp format (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def to_srt(segments: List[Dict]) -> str:
    """Convert segments to SRT format."""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg['start'])
        end = format_timestamp(seg['end'])
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(seg['text'])
        lines.append("")
    return '\n'.join(lines)


def to_vtt(segments: List[Dict]) -> str:
    """Convert segments to WebVTT format."""
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = format_timestamp_vtt(seg['start'])
        end = format_timestamp_vtt(seg['end'])
        lines.append(f"{start} --> {end}")
        lines.append(seg['text'])
        lines.append("")
    return '\n'.join(lines)


def to_txt(segments: List[Dict]) -> str:
    """Convert segments to plain text."""
    return '\n'.join(seg['text'] for seg in segments)


def srt_to_txt(srt_content: str) -> str:
    """
    Convert SRT content to plain text, removing all timestamps and sequence numbers.

    Args:
        srt_content: SRT file content as string

    Returns:
        Plain text without timestamps
    """
    import re
    lines = []
    for line in srt_content.strip().split('\n'):
        line = line.strip()
        # Skip empty lines
        if not line:
            continue
        # Skip sequence numbers (just digits)
        if line.isdigit():
            continue
        # Skip timestamp lines (00:00:00,000 --> 00:00:00,000)
        if re.match(r'\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}', line):
            continue
        lines.append(line)
    return '\n'.join(lines)


def to_json(result: Dict) -> str:
    """Convert result to JSON format."""
    import json
    return json.dumps(result, ensure_ascii=False, indent=2)


def save_transcription(
    result: Dict,
    output_path: str,
    format: str = 'srt'
) -> str:
    """
    Save transcription to file.
    
    Args:
        result: Transcription result from transcribe()
        output_path: Output file path (without extension)
        format: Output format (srt, vtt, txt, json)
    
    Returns:
        Path to saved file
    """
    formatters = {
        'srt': (to_srt, '.srt'),
        'vtt': (to_vtt, '.vtt'),
        'txt': (to_txt, '.txt'),
        'json': (to_json, '.json'),
    }
    
    if format not in formatters:
        console.print(f"[yellow]⚠️  Unknown format '{format}', using 'srt'[/]")
        format = 'srt'
    
    formatter, ext = formatters[format]
    
    # Ensure correct extension
    output_path = str(Path(output_path).with_suffix(ext))
    
    if format == 'json':
        content = formatter(result)
    else:
        content = formatter(result['segments'])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    console.print(f"[bold green]💾 Saved:[/] {output_path}")
    return output_path


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        result = transcribe(sys.argv[1], model_size='tiny')
        print(to_srt(result['segments']))
    else:
        print("Usage: python transcriber.py <audio_file>")
