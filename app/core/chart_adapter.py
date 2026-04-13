from __future__ import annotations

import pandas as pd


REQUIRED = ["timestamp", "open", "high", "low", "close"]
TIMEFRAME_MAP = {
    "1s": None,
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
}


def build_candle_payload(df: pd.DataFrame, timeframe: str = "1s", window: int | None = 300) -> list[dict]:
    if df is None or len(df) == 0:
        return []

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Chart data requires columns: {', '.join(missing)}")

    tf = timeframe if timeframe in TIMEFRAME_MAP else "1s"

    local = df[REQUIRED].copy()
    local["timestamp"] = pd.to_datetime(local["timestamp"], utc=True, errors="coerce")
    local = local.dropna(subset=["timestamp"]).sort_values("timestamp")

    rule = TIMEFRAME_MAP[tf]
    if rule:
        g = local.set_index("timestamp").resample(rule)
        local = pd.DataFrame(
            {
                "open": g["open"].first(),
                "high": g["high"].max(),
                "low": g["low"].min(),
                "close": g["close"].last(),
            }
        ).dropna().reset_index()

    if window is not None:
        local = local.tail(max(20, int(window)))

    out = []
    for _, row in local.iterrows():
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
