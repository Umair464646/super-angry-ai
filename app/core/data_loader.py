from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .schema import REQUIRED_COLUMNS, MINIMAL_COLUMNS


TIMESTAMP_CANDIDATES = [
    "timestamp",
    "open_time",
    "time",
    "datetime",
    "date",
]

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "synthetic",
]

PARQUET_EXTENSIONS = {".parquet", ".pq"}


@dataclass
class DataProfile:
    path: str
    rows: int
    start: str
    end: str
    zero_volume_pct: float
    synthetic_pct: float
    duplicate_timestamps: int
    columns: List[str]
    warnings: List[str]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _normalize_name(name: str) -> str:
    return str(name).strip().lower()


def _find_timestamp_column(columns: List[str]) -> Optional[str]:
    normalized = {_normalize_name(c): c for c in columns}
    for candidate in TIMESTAMP_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _find_available_columns(columns: List[str], wanted: List[str]) -> List[str]:
    wanted_set = {w.lower() for w in wanted}
    return [c for c in columns if _normalize_name(c) in wanted_set]


def _read_csv_header(path: str) -> List[str]:
    header = pd.read_csv(path, nrows=0)
    return [str(c) for c in header.columns]


def _read_parquet_schema_names(path: str) -> List[str]:
    return [str(c) for c in pq.ParquetFile(path).schema.names]


def _resolve_minimal_column_selection(columns: List[str]) -> List[str]:
    timestamp_col = _find_timestamp_column(columns)
    if timestamp_col is None:
        raise ValueError(
            f"Missing timestamp column. Expected one of: {', '.join(TIMESTAMP_CANDIDATES)}"
        )

    selected = [timestamp_col]
    selected.extend(
        c for c in _find_available_columns(columns, MINIMAL_COLUMNS) if c not in selected
    )

    return selected


def load_csv_minimal(path: str) -> pd.DataFrame:
    raw_columns = _read_csv_header(path)
    selected = _resolve_minimal_column_selection(raw_columns)
    return pd.read_csv(path, usecols=selected)


def load_parquet_minimal(path: str) -> pd.DataFrame:
    raw_columns = _read_parquet_schema_names(path)
    selected = _resolve_minimal_column_selection(raw_columns)
    return pd.read_parquet(path, columns=selected)


def _coerce_maybe_epoch(series: pd.Series) -> pd.Series:
    """
    Parse timestamps robustly:
    - datetime strings
    - epoch seconds
    - epoch milliseconds
    - epoch microseconds / nanoseconds if needed
    """
    if series.empty:
        return pd.to_datetime(series, utc=True, errors="coerce")

    # First try generic datetime parsing
    parsed = pd.to_datetime(series, utc=True, errors="coerce")

    # If enough values parsed, keep it
    valid_ratio = float(parsed.notna().mean()) if len(parsed) else 0.0
    if valid_ratio >= 0.8:
        return parsed

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_valid = numeric.notna()

    if not numeric_valid.any():
        return parsed

    sample = numeric[numeric_valid]
    median_abs = float(sample.abs().median()) if not sample.empty else 0.0

    # crude but practical epoch unit detection
    if median_abs >= 1e18:
        unit = "ns"
    elif median_abs >= 1e15:
        unit = "us"
    elif median_abs >= 1e12:
        unit = "ms"
    elif median_abs >= 1e9:
        unit = "s"
    else:
        # very small numbers are unlikely to be real timestamps
        return parsed

    parsed_numeric = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")

    # keep the better parse result
    if parsed_numeric.notna().sum() > parsed.notna().sum():
        return parsed_numeric
    return parsed


def parse_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "timestamp" not in out.columns:
        source_col = _find_timestamp_column(list(out.columns))
        if source_col is None:
            raise ValueError(
                f"Missing timestamp column. Expected one of: {', '.join(TIMESTAMP_CANDIDATES)}"
            )
        if source_col != "timestamp":
            out = out.rename(columns={source_col: "timestamp"})

    out["timestamp"] = _coerce_maybe_epoch(out["timestamp"])

    if out["timestamp"].isna().all():
        raise ValueError("Failed to parse timestamp column into valid UTC datetimes")

    return out


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in NUMERIC_COLUMNS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "synthetic" in out.columns:
        out["synthetic"] = out["synthetic"].fillna(0).astype(int)

    return out


def normalize_ohlc_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Make OHLC internally consistent where possible.
    """
    out = df.copy()
    warnings: List[str] = []

    required = ["open", "high", "low", "close"]
    if not all(c in out.columns for c in required):
        return out, warnings

    # Count weird rows before fixing
    bad_high = (out["high"] < out[["open", "close"]].max(axis=1)).sum()
    bad_low = (out["low"] > out[["open", "close"]].min(axis=1)).sum()
    inverted = (out["high"] < out["low"]).sum()

    if bad_high > 0 or bad_low > 0 or inverted > 0:
        warnings.append(
            "Detected inconsistent OHLC rows. Applied normalization so high/low envelope open/close."
        )

    out["high"] = pd.concat(
        [out["high"], out["open"], out["close"]], axis=1
    ).max(axis=1)

    out["low"] = pd.concat(
        [out["low"], out["open"], out["close"]], axis=1
    ).min(axis=1)

    return out, warnings


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    warnings: List[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, [f"Missing required columns: {', '.join(missing)}"]

    if df.empty:
        return False, ["Loaded dataset is empty"]

    if df["timestamp"].isna().any():
        return False, ["Some timestamps are null after parsing"]

    if df[["open", "high", "low", "close", "volume"]].isna().any().any():
        warnings.append("Some OHLCV values are null")

    if (df["high"] < df["low"]).any():
        warnings.append("Some rows still have high lower than low after normalization")

    if df["timestamp"].duplicated().any():
        warnings.append("Dataset contains duplicate timestamps")

    return True, warnings


def profile_dataframe(df: pd.DataFrame, path: str, warnings: List[str]) -> DataProfile:
    zero_volume_pct = float((df["volume"].fillna(0) == 0).mean() * 100)
    synthetic_pct = (
        float((df["synthetic"].fillna(0) == 1).mean() * 100)
        if "synthetic" in df.columns
        else 0.0
    )
    dupes = int(df["timestamp"].duplicated().sum())

    return DataProfile(
        path=str(path),
        rows=len(df),
        start=str(df["timestamp"].min()),
        end=str(df["timestamp"].max()),
        zero_volume_pct=zero_volume_pct,
        synthetic_pct=synthetic_pct,
        duplicate_timestamps=dupes,
        columns=list(df.columns),
        warnings=warnings,
    )


def _normalize_loaded_dataframe(df: pd.DataFrame, path: str) -> Tuple[pd.DataFrame, DataProfile]:
    df = normalize_columns(df)
    df = parse_timestamp_column(df)
    df = convert_numeric_columns(df)

    ohlc_warnings: List[str] = []
    df, ohlc_warnings = normalize_ohlc_rows(df)

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    ok, validation_warnings = validate_dataframe(df)
    all_warnings = [*ohlc_warnings, *validation_warnings]

    if not ok:
        raise ValueError("; ".join(all_warnings))

    return df, profile_dataframe(df, path, all_warnings)


def _build_arrow_timestamp_filter(table: pa.Table, start=None, end=None):
    timestamp_col_name = _find_timestamp_column(table.column_names)
    if timestamp_col_name is None:
        raise ValueError(
            f"Missing timestamp column. Expected one of: {', '.join(TIMESTAMP_CANDIDATES)}"
        )

    field = table.schema.field(timestamp_col_name)
    col = pc.field(timestamp_col_name)

    def _cast_boundary(value):
        ts = pd.Timestamp(value, tz="UTC")
        if pa.types.is_timestamp(field.type):
            return pa.scalar(ts.to_pydatetime(), type=field.type)
        # fallback for int-based epochs is handled after load, so string/datetime only here
        return pa.scalar(ts.to_pydatetime())

    expr = None

    # Only safe to push down when parquet timestamp column is a timestamp type
    if pa.types.is_timestamp(field.type):
        if start is not None:
            start_expr = col >= _cast_boundary(start)
            expr = start_expr if expr is None else expr & start_expr
        if end is not None:
            end_expr = col <= _cast_boundary(end)
            expr = end_expr if expr is None else expr & end_expr

    return timestamp_col_name, expr


def load_parquet_date_window(path: str, start=None, end=None) -> pd.DataFrame:
    raw_columns = _read_parquet_schema_names(path)
    selected = _resolve_minimal_column_selection(raw_columns)

    parquet_file = pq.ParquetFile(path)
    full_schema = parquet_file.schema_arrow
    empty_table = pa.Table.from_arrays([], schema=full_schema)

    timestamp_col_name, arrow_filter = _build_arrow_timestamp_filter(
        empty_table, start=start, end=end
    )

    # Use pyarrow dataset-style filtered read when possible
    try:
        table = pq.read_table(path, columns=selected, filters=arrow_filter)
        df = table.to_pandas()
    except Exception:
        # fallback path if filter pushdown fails for some schema weirdness
        df = pd.read_parquet(path, columns=selected)

    df = normalize_columns(df)
    if timestamp_col_name != "timestamp" and timestamp_col_name in df.columns:
        df = df.rename(columns={timestamp_col_name: "timestamp"})

    df = parse_timestamp_column(df)
    df = convert_numeric_columns(df)

    if start is not None:
        df = df[df["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df["timestamp"] <= pd.Timestamp(end, tz="UTC")]

    df, _ = normalize_ohlc_rows(df)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def load_market_file_minimal(path: str | Path):
    path = str(path)
    suffix = Path(path).suffix.lower()

    if suffix == ".csv":
        df = load_csv_minimal(path)
    elif suffix in PARQUET_EXTENSIONS:
        df = load_parquet_minimal(path)
    else:
        raise ValueError("Unsupported file type. Use CSV or Parquet.")

    return _normalize_loaded_dataframe(df, path)


def profile_to_text(profile: DataProfile) -> str:
    lines = [
        f"Path: {profile.path}",
        f"Rows: {profile.rows:,}",
        f"Start: {profile.start}",
        f"End: {profile.end}",
        f"Zero-volume bars: {profile.zero_volume_pct:.2f}%",
        f"Synthetic rows: {profile.synthetic_pct:.2f}%",
        f"Duplicate timestamps: {profile.duplicate_timestamps}",
        f"Columns loaded: {', '.join(profile.columns)}",
    ]

    if profile.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {w}" for w in profile.warnings)

    return "\n".join(lines)


def profile_to_dict(profile: DataProfile) -> dict:
    return asdict(profile)