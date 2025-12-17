"""Transcription module using faster-whisper."""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any

from faster_whisper import WhisperModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Available model sizes
MODEL_SIZES = ['tiny', 'base', 'small', 'medium', 'large-v3']


def load_model(
    model_size: str = 'medium',
    device: str = 'auto',
    compute_type: str = 'auto'
) -> WhisperModel:
    """
    Load the Whisper model.
    
    Args:
        model_size: Model size (tiny, base, small, medium, large-v3)
        device: Device to use (auto, cpu, cuda)
        compute_type: Compute type (auto, int8, float16, float32)
    
    Returns:
        Loaded WhisperModel
    """
    if model_size not in MODEL_SIZES:
        console.print(f"[yellow]⚠️  Unknown model size '{model_size}', using 'medium'[/]")
        model_size = 'medium'
    
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
    
    console.print(f"[dim]Device: {device}, Compute type: {compute_type}[/]")
    
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type
    )
    
    console.print(f"[bold green]✅ Model loaded[/]")
    return model


def transcribe(
    audio_path: str,
    model: Optional[WhisperModel] = None,
    model_size: str = 'medium',
    language: Optional[str] = None,
    translate: bool = False,
    word_timestamps: bool = False
) -> Dict[str, Any]:
    """
    Transcribe an audio file.
    
    Args:
        audio_path: Path to the audio file
        model: Pre-loaded WhisperModel (optional)
        model_size: Model size if model not provided
        language: Source language (None for auto-detection)
        translate: If True, translate to English
        word_timestamps: If True, include word-level timestamps
    
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
    
    # Load model if not provided
    if model is None:
        model = load_model(model_size)
    
    console.print(f"[bold blue]🎤 Transcribing:[/] {audio_path}")
    
    task = 'translate' if translate else 'transcribe'
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        progress.add_task(description="Processing audio...", total=None)
        
        segments, info = model.transcribe(
            audio_path,
            language=language,
            task=task,
            word_timestamps=word_timestamps,
            vad_filter=True,  # Filter out silence
        )
        
        # Convert generator to list
        segment_list = []
        full_text = []
        
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
    
    result = {
        'text': ' '.join(full_text),
        'segments': segment_list,
        'language': info.language,
        'language_probability': info.language_probability,
        'duration': info.duration,
    }
    
    console.print(f"[bold green]✅ Transcription complete[/]")
    console.print(f"[dim]Detected language: {info.language} ({info.language_probability:.1%})[/]")
    console.print(f"[dim]Duration: {info.duration:.1f}s, Segments: {len(segment_list)}[/]")
    
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
