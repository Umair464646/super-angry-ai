from __future__ import annotations

import pandas as pd


TIMEFRAME_RULES = {
    "1s": "1s",
    "5s": "5s",
    "15s": "15s",
    "30s": "30s",
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
}


BASE_AGGREGATIONS = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}

OPTIONAL_AGGREGATIONS = {
    "quote_volume": "sum",
    "trades": "sum",
    "buy_volume": "sum",
    "sell_volume": "sum",
    "buy_quote_volume": "sum",
    "sell_quote_volume": "sum",
    "vwap": "mean",
    "avg_trade_size": "mean",
    "buy_sell_vol_delta": "sum",
    "buy_sell_imbalance": "mean",
    "candle_range": "max",
    "body": "mean",
    "upper_wick": "mean",
    "lower_wick": "mean",
    "return_1s": "sum",
    "log_return_1s": "sum",
}


def _validate_input(df: pd.DataFrame) -> None:
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot build timeframe. Missing columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("Cannot build timeframe from empty dataframe")


def _build_agg_map(df: pd.DataFrame) -> dict:
    agg = dict(BASE_AGGREGATIONS)

    for col, rule in OPTIONAL_AGGREGATIONS.items():
        if col in df.columns:
            agg[col] = rule

    return agg


def _aggregate_synthetic(grouped: pd.core.resample.Resampler, has_synthetic: bool) -> pd.Series | None:
    if not has_synthetic:
        return None

    # A resampled bar is synthetic only if all source rows are synthetic.
    synth = grouped["synthetic"].min()
    synth.name = "synthetic"
    return synth.fillna(0).astype(int)


def _post_process_resampled(out: pd.DataFrame) -> pd.DataFrame:
    out = out.dropna(subset=["open", "high", "low", "close"]).copy()

    # normalize OHLC envelope just in case
    out["high"] = out[["high", "open", "close"]].max(axis=1)
    out["low"] = out[["low", "open", "close"]].min(axis=1)

    if "synthetic" in out.columns:
        out["synthetic"] = out["synthetic"].fillna(0).astype(int)

    # keep sorted, clean index
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def build_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    _validate_input(df)

    if timeframe not in TIMEFRAME_RULES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    if timeframe == "1s":
        out = df.copy()
        if "timestamp" in out.columns:
            out = out.sort_values("timestamp").reset_index(drop=True)
        return out

    local = df.copy()
    local["timestamp"] = pd.to_datetime(local["timestamp"], utc=True, errors="coerce")
    local = local.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

    agg_map = _build_agg_map(local)
    grouped = local.resample(TIMEFRAME_RULES[timeframe], label="left", closed="left")

    out = grouped.agg(agg_map)

    synth = _aggregate_synthetic(grouped, "synthetic" in local.columns)
    if synth is not None:
        out["synthetic"] = synth

    out = out.reset_index()
    out = _post_process_resampled(out)
    return out