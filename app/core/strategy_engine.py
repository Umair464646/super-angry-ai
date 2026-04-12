from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import numpy as np

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
    StrategyTemplate(
        key="multi_factor_combo",
        name="Multi-Factor Combo",
        indicators=["EMA", "RSI", "MACD", "ADX", "VWAP", "Volume Spike"],
        params={
            "ema_fast": 20,
            "ema_slow": 50,
            "rsi_len": 14,
            "rsi_long_min": 52,
            "rsi_short_max": 48,
            "adx_min": 18,
            "vol_spike_mult": 1.2,
        },
        entry_logic="Long when trend+momentum+volume align (EMA/MACD/RSI/ADX/VWAP). Short on inverse alignment.",
        exit_logic="Exit via SL/TP/end-of-data in current phase.",
        filters="Synthetic rows do not trigger entries.",
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


def build_strategy_dataframe(df: pd.DataFrame, template_key: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    local = df.copy()
    local = local.sort_values("timestamp").reset_index(drop=True)
    _require_columns(local, ["timestamp", "open", "high", "low", "close", "volume"], template_key)
    p = params or {}

    if template_key == "ema_cross_20_50":
        fast = int(p.get("ema_fast", 20))
        slow = int(p.get("ema_slow", 50))
        if fast <= 1 or slow <= 2 or fast >= slow:
            raise ValueError("EMA parameters invalid; require 1 < ema_fast < ema_slow")
        local["ema_fast"] = local["close"].ewm(span=fast, adjust=False).mean()
        local["ema_slow"] = local["close"].ewm(span=slow, adjust=False).mean()
        local["long_entry"] = (local["ema_fast"] > local["ema_slow"]) & (
            local["ema_fast"].shift(1) <= local["ema_slow"].shift(1)
        )
        local["short_entry"] = (local["ema_fast"] < local["ema_slow"]) & (
            local["ema_fast"].shift(1) >= local["ema_slow"].shift(1)
        )

    elif template_key == "rsi_reversal_30_70":
        rsi_len = int(p.get("rsi_len", 14))
        oversold = float(p.get("oversold", 30))
        overbought = float(p.get("overbought", 70))
        if rsi_len < 2 or not (1 <= oversold < overbought <= 99):
            raise ValueError("RSI parameters invalid")
        delta = local["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(rsi_len).mean()
        avg_loss = loss.rolling(rsi_len).mean().replace(0, pd.NA)
        rs = avg_gain / avg_loss
        local["rsi"] = 100 - (100 / (1 + rs))
        local["long_entry"] = (local["rsi"] > oversold) & (local["rsi"].shift(1) <= oversold)
        local["short_entry"] = (local["rsi"] < overbought) & (local["rsi"].shift(1) >= overbought)

    elif template_key == "breakout_20":
        lookback = int(p.get("lookback", 20))
        if lookback < 2:
            raise ValueError("Breakout lookback invalid")
        local["rolling_high_n"] = local["high"].rolling(lookback).max().shift(1)
        local["rolling_low_n"] = local["low"].rolling(lookback).min().shift(1)
        local["long_entry"] = local["close"] > local["rolling_high_n"]
        local["short_entry"] = local["close"] < local["rolling_low_n"]

    elif template_key == "vwap_reclaim":
        ema_len = int(p.get("ema_len", 34))
        vol_spike_mult = float(p.get("vol_spike_mult", 1.5))
        if ema_len < 2 or vol_spike_mult <= 0:
            raise ValueError("VWAP reclaim parameters invalid")
        local = _ensure_vwap(local)
        local["ema_n"] = local["close"].ewm(span=ema_len, adjust=False).mean()
        vol_mean = local["volume"].rolling(50).mean()
        local["volume_spike"] = local["volume"] > (vol_mean * vol_spike_mult)
        local["long_entry"] = (
            (local["close"] > local["vwap"]) &
            (local["close"].shift(1) <= local["vwap"].shift(1)) &
            (local["close"] > local["ema_n"]) &
            local["volume_spike"]
        )
        local["short_entry"] = (
            (local["close"] < local["vwap"]) &
            (local["close"].shift(1) >= local["vwap"].shift(1)) &
            (local["close"] < local["ema_n"]) &
            local["volume_spike"]
        )

    elif template_key == "multi_factor_combo":
        ema_fast = int(p.get("ema_fast", 20))
        ema_slow = int(p.get("ema_slow", 50))
        rsi_len = int(p.get("rsi_len", 14))
        rsi_long_min = float(p.get("rsi_long_min", 52))
        rsi_short_max = float(p.get("rsi_short_max", 48))
        adx_min = float(p.get("adx_min", 18))
        vol_spike_mult = float(p.get("vol_spike_mult", 1.2))

        if ema_fast <= 1 or ema_slow <= 2 or ema_fast >= ema_slow:
            raise ValueError("multi_factor_combo EMA parameters invalid")

        local = _ensure_vwap(local)
        local["ema_fast"] = local["close"].ewm(span=ema_fast, adjust=False).mean()
        local["ema_slow"] = local["close"].ewm(span=ema_slow, adjust=False).mean()

        delta = local["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(rsi_len).mean()
        avg_loss = loss.rolling(rsi_len).mean().replace(0, pd.NA)
        rs = avg_gain / avg_loss
        local["rsi"] = 100 - (100 / (1 + rs))

        macd_fast = local["close"].ewm(span=12, adjust=False).mean()
        macd_slow = local["close"].ewm(span=26, adjust=False).mean()
        local["macd_line"] = macd_fast - macd_slow
        local["macd_signal"] = local["macd_line"].ewm(span=9, adjust=False).mean()
        local["macd_hist"] = local["macd_line"] - local["macd_signal"]

        plus_dm = (local["high"].diff()).clip(lower=0)
        minus_dm = (-local["low"].diff()).clip(lower=0)
        plus_dm[plus_dm < minus_dm] = 0
        minus_dm[minus_dm < plus_dm] = 0
        tr = pd.concat(
            [
                (local["high"] - local["low"]),
                (local["high"] - local["close"].shift(1)).abs(),
                (local["low"] - local["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean().replace(0, pd.NA)
        plus_di = 100 * (plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA)
        local["adx_14"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

        vol_ma = local["volume"].rolling(50).mean()
        local["volume_spike"] = local["volume"] > (vol_ma * vol_spike_mult)

        trend_up = local["ema_fast"] > local["ema_slow"]
        trend_dn = local["ema_fast"] < local["ema_slow"]
        momentum_up = local["macd_hist"] > 0
        momentum_dn = local["macd_hist"] < 0
        vwap_up = local["close"] > local["vwap"]
        vwap_dn = local["close"] < local["vwap"]
        strong = local["adx_14"] >= adx_min

        local["long_entry"] = trend_up & momentum_up & (local["rsi"] >= rsi_long_min) & strong & vwap_up & local["volume_spike"]
        local["short_entry"] = trend_dn & momentum_dn & (local["rsi"] <= rsi_short_max) & strong & vwap_dn & local["volume_spike"]

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
    params: dict[str, Any] | None = None,
    config: BacktestConfig | None = None,
) -> dict:
    if df is None or df.empty:
        raise ValueError("Dataset is empty")

    staged = build_strategy_dataframe(df, template_key, params=params)
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
    merged_params = dict(template.params)
    if params:
        merged_params.update(params)
    robustness = _robustness_score(train_result.metrics, test_result.metrics)

    return {
        "template": template,
        "params": merged_params,
        "full": full_result,
        "train": train_result,
        "test": test_result,
        "robustness_score": robustness,
    }


def walk_forward_validate(
    df: pd.DataFrame,
    template_key: str,
    params: dict[str, Any] | None = None,
    config: BacktestConfig | None = None,
    folds: int = 4,
) -> tuple[pd.DataFrame, float]:
    if df is None or df.empty:
        raise ValueError("Dataset is empty")

    staged = build_strategy_dataframe(df, template_key, params=params)
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


def _variant_param_grid(template_key: str, base_params: dict[str, Any]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    if template_key == "ema_cross_20_50":
        for fast in [10, 14, 20, 24]:
            for slow in [40, 50, 80]:
                if fast < slow:
                    variants.append({"ema_fast": fast, "ema_slow": slow})
    elif template_key == "rsi_reversal_30_70":
        for rsi_len in [7, 14, 21]:
            for oversold, overbought in [(25, 75), (30, 70), (35, 65)]:
                variants.append({"rsi_len": rsi_len, "oversold": oversold, "overbought": overbought})
    elif template_key == "breakout_20":
        for lookback in [10, 20, 30, 55]:
            variants.append({"lookback": lookback})
    elif template_key == "vwap_reclaim":
        for ema_len in [21, 34, 55]:
            for vsm in [1.2, 1.5, 2.0]:
                variants.append({"ema_len": ema_len, "vol_spike_mult": vsm})
    elif template_key == "multi_factor_combo":
        for ema_fast in [12, 20]:
            for ema_slow in [34, 50, 80]:
                if ema_fast >= ema_slow:
                    continue
                for rsi_len in [10, 14]:
                    for rsi_long_min, rsi_short_max in [(55, 45), (52, 48)]:
                        for adx_min in [16, 20, 24]:
                            for vsm in [1.1, 1.3]:
                                variants.append(
                                    {
                                        "ema_fast": ema_fast,
                                        "ema_slow": ema_slow,
                                        "rsi_len": rsi_len,
                                        "rsi_long_min": rsi_long_min,
                                        "rsi_short_max": rsi_short_max,
                                        "adx_min": adx_min,
                                        "vol_spike_mult": vsm,
                                    }
                                )
    else:
        variants.append(base_params)
    return variants or [base_params]


def evolve_templates(
    df: pd.DataFrame,
    config: BacktestConfig | None = None,
    top_k: int = 8,
    progress_cb=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or BacktestConfig()
    all_rows: list[dict[str, Any]] = []

    grids = [(t, p) for t in TEMPLATES for p in _variant_param_grid(t.key, t.params)]
    total = max(1, len(grids))

    for idx, (t, params) in enumerate(grids, start=1):
        if progress_cb is not None and (idx == 1 or idx % 3 == 0 or idx == total):
            try:
                progress_cb(idx, total, t.name)
            except Exception:
                pass
        try:
            evaluated = evaluate_template(df, t.key, params=params, config=cfg)
            test = evaluated["test"].metrics
            all_rows.append(
                {
                    "strategy": t.name,
                    "template_key": t.key,
                    "params": evaluated["params"],
                    "robustness_score": float(evaluated["robustness_score"]),
                    "test_return_pct": float(test["total_return_pct"]),
                    "test_win_rate_pct": float(test["win_rate_pct"]),
                    "test_max_drawdown_pct": float(test["max_drawdown_pct"]),
                    "test_trades": int(test["total_trades"]),
                }
            )
        except Exception:
            continue

    if not all_rows:
        raise ValueError("Evolution engine could not evaluate any template variants")

    frame = pd.DataFrame(all_rows)
    frame["fitness"] = (
        frame["robustness_score"] * 0.50 +
        frame["test_return_pct"] * 0.30 +
        frame["test_win_rate_pct"] * 0.10 -
        frame["test_max_drawdown_pct"].abs() * 0.10
    )
    frame = frame.sort_values("fitness", ascending=False).reset_index(drop=True)
    top = frame.head(max(1, int(top_k))).reset_index(drop=True)
    return frame, top


def tradingview_strategy_text(template_key: str, params: dict[str, Any]) -> str:
    template = _template_by_key(template_key)
    p = ", ".join(f"{k}={v}" for k, v in params.items())
    return (
        f"Strategy: {template.name}\n"
        f"Template Key: {template_key}\n"
        f"Parameters: {p}\n"
        f"Entry Logic: {template.entry_logic}\n"
        f"Exit Logic: {template.exit_logic}\n"
        f"Filters: {template.filters}\n"
        "Replicable TradingView Notes:\n"
        "- Use identical indicator lengths/thresholds in Pine.\n"
        "- Keep next-bar execution assumptions aligned with backtest settings.\n"
        "- Disable entries on synthetic/no-trade bars where possible.\n"
    )
