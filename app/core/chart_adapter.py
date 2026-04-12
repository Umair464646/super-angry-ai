from __future__ import annotations

import pandas as pd


REQUIRED = ["timestamp", "open", "high", "low", "close"]


def build_candle_payload(df: pd.DataFrame, timeframe: str = "1s", window: int = 300) -> list[dict]:
    if df is None or len(df) == 0:
        return []

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Chart data requires columns: {', '.join(missing)}")

    local = df[REQUIRED].copy()
    local["timestamp"] = pd.to_datetime(local["timestamp"], utc=True, errors="coerce")
    local = local.dropna(subset=["timestamp"]).sort_values("timestamp")

    if timeframe == "1m":
        g = local.set_index("timestamp").resample("1min")
        local = pd.DataFrame(
            {
                "open": g["open"].first(),
                "high": g["high"].max(),
                "low": g["low"].min(),
                "close": g["close"].last(),
            }
        ).dropna().reset_index()

    tail = local.tail(max(20, int(window)))
    out = []
    for _, row in tail.iterrows():
        out.append(
            {
                "t": str(row["timestamp"]),
                "o": float(row["open"]),
                "h": float(row["high"]),
                "l": float(row["low"]),
                "c": float(row["close"]),
            }
        )
    return out
