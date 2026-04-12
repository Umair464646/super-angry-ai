from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.core.backtest_engine import BacktestConfig, run_backtest


@dataclass(frozen=True)
class StrategyTemplate:
    key: str
    name: str
    indicators: list[str]
    params: dict[str, Any]
    entry_logic: str
    exit_logic: str
    filters: str


TEMPLATES: list[StrategyTemplate] = [
    StrategyTemplate(
        key="ema_cross_20_50",
        name="EMA Cross 20/50",
        indicators=["EMA(20)", "EMA(50)"],
        params={"ema_fast": 20, "ema_slow": 50},
        entry_logic="Long: EMA20 crosses above EMA50. Short: EMA20 crosses below EMA50.",
        exit_logic="Exit via stop-loss, take-profit, or end of data in current phase.",
        filters="Ignore synthetic rows for signal triggers.",
    ),
    StrategyTemplate(
        key="rsi_reversal_30_70",
        name="RSI Reversal 30/70",
        indicators=["RSI(14)"],
        params={"rsi_len": 14, "oversold": 30, "overbought": 70},
        entry_logic="Long on RSI crossing above 30. Short on RSI crossing below 70.",
        exit_logic="Exit via stop-loss, take-profit, or end of data in current phase.",
        filters="Ignore synthetic rows for signal triggers.",
    ),
    StrategyTemplate(
        key="breakout_20",
        name="Breakout 20",
        indicators=["Rolling High(20)", "Rolling Low(20)"],
        params={"lookback": 20},
        entry_logic="Long when close breaks previous 20-bar high. Short when close breaks previous 20-bar low.",
        exit_logic="Exit via stop-loss, take-profit, or end of data in current phase.",
        filters="Ignore synthetic rows for signal triggers.",
    ),
    StrategyTemplate(
        key="vwap_reclaim",
        name="VWAP Reclaim",
        indicators=["VWAP", "EMA(34)", "Volume Spike"],
        params={"ema_len": 34, "vol_spike_mult": 1.5},
        entry_logic="Long when close reclaims above VWAP + EMA34 trend confirmation + volume spike. Short inverse.",
        exit_logic="Exit via stop-loss, take-profit, or end of data in current phase.",
        filters="Ignore synthetic rows and require valid VWAP values.",
    ),
]


def _template_by_key(key: str) -> StrategyTemplate:
    for t in TEMPLATES:
        if t.key == key:
            return t
    raise ValueError(f"Unknown strategy template: {key}")


def _require_columns(df: pd.DataFrame, cols: list[str], template_key: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Template '{template_key}' requires missing columns: {', '.join(missing)}"
        )


def _ensure_vwap(local: pd.DataFrame) -> pd.DataFrame:
    if "vwap" in local.columns and local["vwap"].notna().any():
        return local

    if "quote_volume" in local.columns:
        cum_quote = local["quote_volume"].fillna(0).cumsum()
        cum_volume = local["volume"].fillna(0).replace(0, pd.NA).cumsum()
        local["vwap"] = (cum_quote / cum_volume).ffill()
        return local

    typical_price = (local["high"] + local["low"] + local["close"]) / 3.0
    pv = (typical_price * local["volume"].fillna(0)).cumsum()
    vv = local["volume"].fillna(0).replace(0, pd.NA).cumsum()
    local["vwap"] = (pv / vv).ffill()
    return local


def build_strategy_dataframe(df: pd.DataFrame, template_key: str) -> pd.DataFrame:
    local = df.copy()
    local = local.sort_values("timestamp").reset_index(drop=True)
    _require_columns(local, ["timestamp", "open", "high", "low", "close", "volume"], template_key)

    if template_key == "ema_cross_20_50":
        local["ema_fast"] = local["close"].ewm(span=20, adjust=False).mean()
        local["ema_slow"] = local["close"].ewm(span=50, adjust=False).mean()
        local["long_entry"] = (local["ema_fast"] > local["ema_slow"]) & (
            local["ema_fast"].shift(1) <= local["ema_slow"].shift(1)
        )
        local["short_entry"] = (local["ema_fast"] < local["ema_slow"]) & (
            local["ema_fast"].shift(1) >= local["ema_slow"].shift(1)
        )

    elif template_key == "rsi_reversal_30_70":
        delta = local["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean().replace(0, pd.NA)
        rs = avg_gain / avg_loss
        local["rsi"] = 100 - (100 / (1 + rs))
        local["long_entry"] = (local["rsi"] > 30) & (local["rsi"].shift(1) <= 30)
        local["short_entry"] = (local["rsi"] < 70) & (local["rsi"].shift(1) >= 70)

    elif template_key == "breakout_20":
        local["rolling_high_20"] = local["high"].rolling(20).max().shift(1)
        local["rolling_low_20"] = local["low"].rolling(20).min().shift(1)
        local["long_entry"] = local["close"] > local["rolling_high_20"]
        local["short_entry"] = local["close"] < local["rolling_low_20"]

    elif template_key == "vwap_reclaim":
        local = _ensure_vwap(local)
        local["ema_34"] = local["close"].ewm(span=34, adjust=False).mean()
        vol_mean = local["volume"].rolling(50).mean()
        local["volume_spike"] = local["volume"] > (vol_mean * 1.5)
        local["long_entry"] = (
            (local["close"] > local["vwap"]) &
            (local["close"].shift(1) <= local["vwap"].shift(1)) &
            (local["close"] > local["ema_34"]) &
            local["volume_spike"]
        )
        local["short_entry"] = (
            (local["close"] < local["vwap"]) &
            (local["close"].shift(1) >= local["vwap"].shift(1)) &
            (local["close"] < local["ema_34"]) &
            local["volume_spike"]
        )

    else:
        raise ValueError(f"Unsupported template: {template_key}")

    local["long_entry"] = local["long_entry"].fillna(False)
    local["short_entry"] = local["short_entry"].fillna(False)

    if "synthetic" in local.columns:
        mask = local["synthetic"].fillna(0).astype(int) == 1
        local.loc[mask, "long_entry"] = False
        local.loc[mask, "short_entry"] = False

    return local


def _robustness_score(train_metrics: dict, test_metrics: dict) -> float:
    ret_gap = abs(float(train_metrics["total_return_pct"]) - float(test_metrics["total_return_pct"]))
    win_gap = abs(float(train_metrics["win_rate_pct"]) - float(test_metrics["win_rate_pct"]))
    dd_penalty = max(0.0, -float(test_metrics["max_drawdown_pct"]))
    trades = float(test_metrics["total_trades"])
    trade_bonus = min(15.0, trades / 4.0)

    score = 100.0 - (0.8 * ret_gap) - (0.5 * win_gap) - (0.6 * dd_penalty) + trade_bonus
    return round(max(0.0, min(100.0, score)), 2)


def evaluate_template(
    df: pd.DataFrame,
    template_key: str,
    config: BacktestConfig | None = None,
) -> dict:
    if df is None or df.empty:
        raise ValueError("Dataset is empty")

    staged = build_strategy_dataframe(df, template_key)
    n = len(staged)
    split = max(200, int(n * 0.7))
    split = min(split, n - 50)
    if split <= 100:
        raise ValueError("Not enough rows for train/test split validation")

    train_df = staged.iloc[:split].reset_index(drop=True)
    test_df = staged.iloc[split:].reset_index(drop=True)

    cfg = config or BacktestConfig()

    full_result = run_backtest(staged, cfg)
    train_result = run_backtest(train_df, cfg)
    test_result = run_backtest(test_df, cfg)

    template = _template_by_key(template_key)
    robustness = _robustness_score(train_result.metrics, test_result.metrics)

    return {
        "template": template,
        "full": full_result,
        "train": train_result,
        "test": test_result,
        "robustness_score": robustness,
    }


def walk_forward_validate(
    df: pd.DataFrame,
    template_key: str,
    config: BacktestConfig | None = None,
    folds: int = 4,
) -> tuple[pd.DataFrame, float]:
    if df is None or df.empty:
        raise ValueError("Dataset is empty")

    staged = build_strategy_dataframe(df, template_key)
    cfg = config or BacktestConfig()

    fold_size = max(100, len(staged) // (folds + 1))
    rows = []

    for i in range(folds):
        start = i * fold_size
        end = min(len(staged), start + fold_size)
        if end - start < 50:
            continue

        fold_df = staged.iloc[start:end].reset_index(drop=True)
        result = run_backtest(fold_df, cfg)
        m = result.metrics
        rows.append(
            {
                "fold": i + 1,
                "rows": len(fold_df),
                "return_pct": float(m["total_return_pct"]),
                "trades": int(m["total_trades"]),
                "win_rate_pct": float(m["win_rate_pct"]),
                "max_drawdown_pct": float(m["max_drawdown_pct"]),
            }
        )

    if not rows:
        raise ValueError("Could not build validation folds from this dataset")

    frame = pd.DataFrame(rows)
    stability = 100.0
    stability -= frame["return_pct"].std(ddof=0) * 0.8 if len(frame) > 1 else 0.0
    stability -= abs(frame["max_drawdown_pct"].mean()) * 0.5
    stability += min(10.0, frame["trades"].mean() / 5.0)
    stability = round(max(0.0, min(100.0, stability)), 2)

    return frame, stability
