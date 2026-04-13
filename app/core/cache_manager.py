from __future__ import annotations
import hashlib, json
from pathlib import Path

def dataset_cache_dir(source_path: str) -> Path:
    p = Path(source_path)
    stamp = f"{p.resolve()}|{p.stat().st_mtime_ns}|{p.stat().st_size}"
    digest = hashlib.sha1(stamp.encode("utf-8")).hexdigest()[:16]
    base = p.parent / ".crypto_lab_cache" / f"{p.stem}_{digest}"
    base.mkdir(parents=True, exist_ok=True)
    return base

def timeframe_cache_path(source_path: str, timeframe: str) -> Path:
    return dataset_cache_dir(source_path) / f"{timeframe}.parquet"

def profile_cache_path(source_path: str) -> Path:
    return dataset_cache_dir(source_path) / "profile.json"

def feature_export_dir(source_path: str) -> Path:
    p = dataset_cache_dir(source_path) / "features"
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_profile_cache(source_path: str, payload: dict) -> None:
    profile_cache_path(source_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
