"""System dependency checker and auto-installer."""

import os
import platform
import shutil
import subprocess
import sys

from rich.console import Console
from rich.panel import Panel

console = Console()


def check_dependency(name: str) -> bool:
    """Check if a system dependency is available."""
    return shutil.which(name) is not None


def has_brew() -> bool:
    """Check if Homebrew is installed."""
    return shutil.which('brew') is not None


def install_with_brew(package: str) -> bool:
    """Install a package using Homebrew."""
    try:
        console.print(f"[bold blue]Installing {package} with Homebrew...[/]")
        result = subprocess.run(
            ['brew', 'install', package],
            capture_output=False,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def install_deno_script() -> bool:
    """Install Deno using the official install script."""
    system = platform.system()
    try:
        console.print("[bold blue]Installing Deno...[/]")
        if system == 'Windows':
            subprocess.run(
                ['powershell', '-c', 'irm https://deno.land/install.ps1 | iex'],
                check=True,
            )
        else:
            subprocess.run(
                ['sh', '-c', 'curl -fsSL https://deno.land/install.sh | sh'],
                check=True,
            )
        return True
    except subprocess.CalledProcessError:
        return False


def install_ffmpeg() -> bool:
    """Try to auto-install ffmpeg."""
    system = platform.system()

    if system == 'Darwin' and has_brew():
        return install_with_brew('ffmpeg')

    # Can't auto-install, show manual instructions
    return False


def install_deno() -> bool:
    """Try to auto-install deno."""
    system = platform.system()

    if system == 'Darwin' and has_brew():
        return install_with_brew('deno')

    # Try install script for Linux/macOS without brew
    if system in ('Darwin', 'Linux'):
        return install_deno_script()

    # Can't auto-install on Windows without manual steps
    return False


def install_mlx() -> bool:
    """Try to auto-install mlx-whisper."""
    try:
        console.print("[bold blue]Installing mlx-whisper...[/]")
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', 'mlx-whisper'],
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def get_manual_install_command(tool: str) -> str:
    """Get manual install command for a tool."""
    system = platform.system()

    commands = {
        'ffmpeg': {
            'Darwin': 'brew install ffmpeg',
            'Linux': 'sudo apt install ffmpeg  # or: sudo dnf install ffmpeg',
            'Windows': 'choco install ffmpeg  # or download from https://ffmpeg.org',
        },
        'deno': {
            'Darwin': 'brew install deno  # or: curl -fsSL https://deno.land/install.sh | sh',
            'Linux': 'curl -fsSL https://deno.land/install.sh | sh',
            'Windows': 'irm https://deno.land/install.ps1 | iex',
        },
    }

    return commands.get(tool, {}).get(system, f'Please install {tool} manually')


def ensure_dependency(name: str, installer_func, reason: str) -> bool:
    """Ensure a dependency is installed, try auto-install if missing."""
    if check_dependency(name):
        return True

    console.print(f"\n[yellow]⚠️  {name} not found[/] - {reason}")

    # Try auto-install
    if installer_func():
        # Verify installation
        if check_dependency(name):
            console.print(f"[green]✅ {name} installed successfully![/]")
            return True

    # Auto-install failed, show manual instructions
    cmd = get_manual_install_command(name)
    console.print(f"[red]Could not auto-install {name}.[/]")
    console.print(f"Please install manually: [cyan]{cmd}[/]\n")
    return False


def check_and_install_dependencies(need_deno: bool = False):
    """Check and auto-install required system dependencies."""
    all_ok = True

    # FFmpeg is always required
    if not ensure_dependency('ffmpeg', install_ffmpeg, 'Required for audio processing'):
        all_ok = False

    # Deno is only needed for YouTube
    if need_deno:
        if not ensure_dependency('deno', install_deno, 'Required for YouTube downloads'):
            all_ok = False

    if not all_ok:
        sys.exit(1)


def check_and_install_mlx():
    """Check if MLX is recommended but not installed, offer to install."""
    # Only on Apple Silicon
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        return

    # Check if mlx-whisper is installed
    try:
        import mlx_whisper
        return  # Already installed
    except ImportError:
        pass

    console.print()
    console.print(Panel(
        "[bold yellow]MLX acceleration available![/]\n\n"
        "You're on Apple Silicon. Installing mlx-whisper for 2-3x faster transcription...",
        title="💡 Optimizing",
        border_style="yellow"
    ))

    if install_mlx():
        console.print("[green]✅ mlx-whisper installed! Restart to use GPU acceleration.[/]\n")
    else:
        console.print("[yellow]Could not install mlx-whisper. Run manually:[/]")
        console.print("[cyan]pip install mlx-whisper[/]\n")


# Backwards compatibility aliases
def check_and_exit_if_missing(need_deno: bool = False):
    """Backwards compatible alias."""
    check_and_install_dependencies(need_deno=need_deno)


def check_mlx_recommendation():
    """Backwards compatible alias."""
    check_and_install_mlx()
