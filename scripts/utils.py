"""
ThermalSense AI — shared utilities
Person A's helper module used by all pipeline scripts.
"""

import os
import sys
import time
import yaml
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from functools import wraps

import numpy as np
from loguru import logger
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text

# ─── Setup ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

console = Console()


def setup_logger(script_name: str) -> None:
    """Configure loguru to write to both terminal and log file."""
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{script_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
        colorize=True,
    )
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}",
        level="DEBUG",
        rotation="50 MB",
    )
    logger.info(f"Logging to {log_file}")


def load_config() -> dict:
    """Load pipeline config from config/config.yaml."""
    config_path = ROOT / "config" / "config.yaml"
    if not config_path.exists():
        logger.error(f"Config not found at {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    logger.debug(f"Config loaded from {config_path}")
    return cfg


def get_city_config(city_name: str) -> dict:
    """Get config for a specific city, with helpful error on unknown city."""
    cfg = load_config()
    cities = cfg.get("cities", {})
    if city_name not in cities:
        available = list(cities.keys())
        logger.error(f"City '{city_name}' not in config. Available: {available}")
        sys.exit(1)
    return {**cities[city_name], **{"pipeline": cfg["pipeline"], "paths": cfg["paths"]}}


def make_progress() -> Progress:
    """Consistent progress bar used across all scripts."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )


def print_banner(title: str, subtitle: str = "") -> None:
    """Print a rich banner at the start of each script."""
    text = Text(title, style="bold white")
    if subtitle:
        text.append(f"\n{subtitle}", style="dim white")
    console.print(Panel(text, style="blue", padding=(0, 2)))


def retry(max_attempts: int = 3, wait_seconds: float = 5.0, exceptions=(Exception,)):
    """Decorator: retry a function up to max_attempts times on exception."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(f"{fn.__name__} failed after {max_attempts} attempts: {e}")
                        raise
                    logger.warning(f"{fn.__name__} attempt {attempt}/{max_attempts} failed: {e}. Retrying in {wait_seconds}s...")
                    time.sleep(wait_seconds)
        return wrapper
    return decorator


def ensure_dirs(*paths: Path) -> None:
    """Create directories if they don't exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def file_checksum(filepath: Path) -> str:
    """MD5 checksum of a file — used for caching to skip re-downloads."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def save_metadata(output_path: Path, metadata: dict) -> None:
    """Save a JSON sidecar file next to any output — for reproducibility."""
    meta_path = output_path.with_suffix(".meta.json")
    metadata["generated_at"] = datetime.now(timezone.utc).isoformat()
    metadata["script_version"] = "1.0.0"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.debug(f"Metadata saved to {meta_path}")


def bbox_to_ee_geometry(bbox: list):
    """Convert [west, south, east, north] bbox to ee.Geometry.Rectangle."""
    import ee
    west, south, east, north = bbox
    return ee.Geometry.Rectangle([west, south, east, north])


def kelvin_to_celsius(arr: np.ndarray) -> np.ndarray:
    """Convert Kelvin array to Celsius."""
    return arr - 273.15


def normalize_0_1(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize array to [0, 1], ignoring NaN."""
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx == mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def log_raster_stats(arr: np.ndarray, name: str, unit: str = "") -> None:
    """Log basic statistics for a raster array — sanity check."""
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        logger.warning(f"{name}: ALL VALUES ARE NaN — check your inputs!")
        return
    logger.info(
        f"{name}: min={valid.min():.2f} max={valid.max():.2f} "
        f"mean={valid.mean():.2f} std={valid.std():.2f} "
        f"nan%={(np.isnan(arr).sum() / arr.size * 100):.1f}%"
        + (f" [{unit}]" if unit else "")
    )
