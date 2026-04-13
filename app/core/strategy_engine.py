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
    StrategyTemplate(
        key="adaptive_indicator_mesh",
        name="Adaptive Indicator Mesh",
        indicators=["EMA", "RSI", "MACD", "ADX", "VWAP", "Bollinger", "Stoch", "CCI", "Williams %R", "CMF", "OBV"],
        params={
            "ema_fast": 12,
            "ema_slow": 55,
            "rsi_len": 14,
            "rsi_long_min": 54,
            "rsi_short_max": 46,
            "adx_min": 18,
            "vol_spike_mult": 1.15,
            "bb_len": 20,
            "stoch_len": 14,
            "cci_len": 20,
            "williams_len": 14,
            "cmf_len": 20,
            "use_vwap": 1,
            "use_bbands": 1,
            "use_stoch": 1,
            "use_cci": 1,
            "use_williams": 1,
            "use_cmf": 1,
            "use_obv": 1,
        },
        entry_logic="Weighted multi-indicator voting with adaptive thresholds; requires minimum votes from enabled indicators.",
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
    elif template_key == "adaptive_indicator_mesh":
        ema_fast = int(p.get("ema_fast", 12))
        ema_slow = int(p.get("ema_slow", 55))
        rsi_len = int(p.get("rsi_len", 14))
        rsi_long_min = float(p.get("rsi_long_min", 54))
        rsi_short_max = float(p.get("rsi_short_max", 46))
        adx_min = float(p.get("adx_min", 18))
        vol_spike_mult = float(p.get("vol_spike_mult", 1.15))
        bb_len = int(p.get("bb_len", 20))
        stoch_len = int(p.get("stoch_len", 14))
        cci_len = int(p.get("cci_len", 20))
        williams_len = int(p.get("williams_len", 14))
        cmf_len = int(p.get("cmf_len", 20))
        use_vwap = int(p.get("use_vwap", 1))
        use_bbands = int(p.get("use_bbands", 1))
        use_stoch = int(p.get("use_stoch", 1))
        use_cci = int(p.get("use_cci", 1))
        use_williams = int(p.get("use_williams", 1))
        use_cmf = int(p.get("use_cmf", 1))
        use_obv = int(p.get("use_obv", 1))
        if ema_fast <= 1 or ema_slow <= 2 or ema_fast >= ema_slow:
            raise ValueError("adaptive_indicator_mesh EMA parameters invalid")

        local = _ensure_vwap(local)
        local["ema_fast"] = local["close"].ewm(span=ema_fast, adjust=False).mean()
        local["ema_slow"] = local["close"].ewm(span=ema_slow, adjust=False).mean()
        delta = local["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(rsi_len).mean()
        avg_loss = loss.rolling(rsi_len).mean().replace(0, pd.NA)
        local["rsi"] = 100 - (100 / (1 + (avg_gain / avg_loss)))

        macd_fast = local["close"].ewm(span=12, adjust=False).mean()
        macd_slow = local["close"].ewm(span=26, adjust=False).mean()
        local["macd_line"] = macd_fast - macd_slow
        local["macd_signal"] = local["macd_line"].ewm(span=9, adjust=False).mean()
        local["macd_hist"] = local["macd_line"] - local["macd_signal"]

        plus_dm = (local["high"].diff()).clip(lower=0)
        minus_dm = (-local["low"].diff()).clip(lower=0)
        plus_dm[plus_dm < minus_dm] = 0
        minus_dm[minus_dm < plus_dm] = 0
        tr = pd.concat([(local["high"] - local["low"]), (local["high"] - local["close"].shift(1)).abs(), (local["low"] - local["close"].shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean().replace(0, pd.NA)
        plus_di = 100 * (plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA)
        local["adx_14"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

        vol_ma = local["volume"].rolling(50).mean()
        local["volume_spike"] = local["volume"] > (vol_ma * vol_spike_mult)

        # Optional indicator blocks
        bb_mid = local["close"].rolling(bb_len).mean()
        bb_std = local["close"].rolling(bb_len).std(ddof=0)
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std

        stoch_low = local["low"].rolling(stoch_len).min()
        stoch_high = local["high"].rolling(stoch_len).max()
        stoch_k = ((local["close"] - stoch_low) / (stoch_high - stoch_low).replace(0, pd.NA)) * 100.0

        tp = (local["high"] + local["low"] + local["close"]) / 3.0
        cci_ma = tp.rolling(cci_len).mean()
        cci_md = (tp - cci_ma).abs().rolling(cci_len).mean().replace(0, pd.NA)
        cci = (tp - cci_ma) / (0.015 * cci_md)

        will_hh = local["high"].rolling(williams_len).max()
        will_ll = local["low"].rolling(williams_len).min()
        willr = -100 * ((will_hh - local["close"]) / (will_hh - will_ll).replace(0, pd.NA))

        mfm = ((local["close"] - local["low"]) - (local["high"] - local["close"])) / (local["high"] - local["low"]).replace(0, pd.NA)
        cmf = (mfm * local["volume"]).rolling(cmf_len).sum() / local["volume"].rolling(cmf_len).sum().replace(0, pd.NA)

        obv = (np.sign(local["close"].diff().fillna(0)) * local["volume"]).cumsum()
        obv_up = obv > obv.ewm(span=21, adjust=False).mean()

        long_votes = (
            (local["ema_fast"] > local["ema_slow"]).astype(int)
            + (local["macd_hist"] > 0).astype(int)
            + (local["rsi"] >= rsi_long_min).astype(int)
            + (local["adx_14"] >= adx_min).astype(int)
            + (local["volume_spike"]).astype(int)
            + (use_vwap * (local["close"] > local["vwap"]).astype(int))
            + (use_bbands * (local["close"] <= bb_upper).astype(int))
            + (use_stoch * (stoch_k > 55).astype(int))
            + (use_cci * (cci > 0).astype(int))
            + (use_williams * (willr > -50).astype(int))
            + (use_cmf * (cmf > 0).astype(int))
            + (use_obv * obv_up.astype(int))
        )
        short_votes = (
            (local["ema_fast"] < local["ema_slow"]).astype(int)
            + (local["macd_hist"] < 0).astype(int)
            + (local["rsi"] <= rsi_short_max).astype(int)
            + (local["adx_14"] >= adx_min).astype(int)
            + (local["volume_spike"]).astype(int)
            + (use_vwap * (local["close"] < local["vwap"]).astype(int))
            + (use_bbands * (local["close"] >= bb_lower).astype(int))
            + (use_stoch * (stoch_k < 45).astype(int))
            + (use_cci * (cci < 0).astype(int))
            + (use_williams * (willr < -50).astype(int))
            + (use_cmf * (cmf < 0).astype(int))
            + (use_obv * (~obv_up).astype(int))
        )
        enabled_optional = use_vwap + use_bbands + use_stoch + use_cci + use_williams + use_cmf + use_obv
        min_votes = max(5, 4 + int(enabled_optional * 0.6))
        local["long_entry"] = long_votes >= min_votes
        local["short_entry"] = short_votes >= min_votes

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


def _performance_context_analysis(test_df: pd.DataFrame, trades_df: pd.DataFrame) -> dict[str, Any]:
    def _empty() -> dict[str, Any]:
        return {
            "high_vol_avg_return": 0.0,
            "low_vol_avg_return": 0.0,
            "trending_avg_return": 0.0,
            "ranging_avg_return": 0.0,
            "trend_confidence": 0.0,
            "volatility_confidence": 0.0,
            "context_confidence": 0.0,
            "time_stability": 0.0,
            "decay_score": 0.0,
            "decay_flag": False,
            "sample_count": 0,
            "performance_context": "Insufficient trades for context analysis",
        }

    def _condition_confidence(a: pd.Series, b: pd.Series) -> float:
        a = pd.to_numeric(a, errors="coerce").dropna()
        b = pd.to_numeric(b, errors="coerce").dropna()
        n1, n2 = len(a), len(b)
        if n1 < 2 or n2 < 2:
            return 0.0
        total = n1 + n2
        p = n1 / total
        coverage = 4.0 * p * (1.0 - p)  # 0..1, max at balanced samples
        m1, m2 = float(a.mean()), float(b.mean())
        v1, v2 = float(a.var(ddof=1)), float(b.var(ddof=1))
        pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / max(1, (n1 + n2 - 2))
        pooled_std = float(np.sqrt(max(pooled, 1e-12)))
        effect = abs(m1 - m2) / pooled_std
        magnitude = effect / (1.0 + effect)
        n_eff = (n1 * n2) / max(1.0, (n1 + n2))
        strength = 1.0 - float(np.exp(-np.sqrt(max(n_eff, 0.0)) * effect))
        return float(max(0.0, min(1.0, coverage * magnitude * strength)))

    def _consistency_score(values: list[float]) -> float:
        arr = np.asarray(values, dtype=float)
        if len(arr) <= 1:
            return 0.0
        mean_abs = float(np.mean(np.abs(arr)))
        sd = float(np.std(arr, ddof=0))
        cv = sd / (mean_abs + 1e-9)
        return float(1.0 / (1.0 + cv))

    if test_df is None or test_df.empty or trades_df is None or trades_df.empty:
        return _empty()

    local = test_df.copy()
    local["timestamp"] = pd.to_datetime(local["timestamp"], utc=True, errors="coerce")
    local = local.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    local["ret"] = local["close"].pct_change().fillna(0.0)
    local["volatility"] = local["ret"].rolling(20).std(ddof=0).fillna(0.0)
    ema_fast = local["close"].ewm(span=20, adjust=False).mean()
    ema_slow = local["close"].ewm(span=50, adjust=False).mean()
    local["trend_strength"] = ((ema_fast - ema_slow).abs() / local["close"].replace(0, pd.NA)).fillna(0.0)

    vol_thr = float(local["volatility"].median())
    trend_thr = float(local["trend_strength"].median())

    trades = trades_df.copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
    trades = trades.dropna(subset=["entry_time"]).sort_values("entry_time").reset_index(drop=True)
    if trades.empty:
        return _empty()

    tagged = pd.merge_asof(
        trades,
        local[["timestamp", "volatility", "trend_strength"]].sort_values("timestamp"),
        left_on="entry_time",
        right_on="timestamp",
        direction="backward",
    )
    tagged["return_pct"] = pd.to_numeric(tagged["return_pct"], errors="coerce").fillna(0.0)
    tagged["high_vol"] = tagged["volatility"] >= vol_thr
    tagged["trending"] = tagged["trend_strength"] >= trend_thr

    hv = float(tagged.loc[tagged["high_vol"], "return_pct"].mean()) if (tagged["high_vol"]).any() else 0.0
    lv = float(tagged.loc[~tagged["high_vol"], "return_pct"].mean()) if (~tagged["high_vol"]).any() else 0.0
    tr = float(tagged.loc[tagged["trending"], "return_pct"].mean()) if (tagged["trending"]).any() else 0.0
    rg = float(tagged.loc[~tagged["trending"], "return_pct"].mean()) if (~tagged["trending"]).any() else 0.0
    n_high = int(tagged["high_vol"].sum())
    n_low = int((~tagged["high_vol"]).sum())
    n_tr = int(tagged["trending"].sum())
    n_rg = int((~tagged["trending"]).sum())
    trend_conf = _condition_confidence(
        tagged.loc[tagged["trending"], "return_pct"],
        tagged.loc[~tagged["trending"], "return_pct"],
    )
    vol_conf = _condition_confidence(
        tagged.loc[tagged["high_vol"], "return_pct"],
        tagged.loc[~tagged["high_vol"], "return_pct"],
    )
    context_conf = float(np.sqrt(max(0.0, trend_conf * vol_conf)))
    return_scale = float(np.std(tagged["return_pct"], ddof=0))

    # Time-segment stability and decay (2-4 adaptive segments).
    seg_count = int(min(4, max(2, np.sqrt(max(4, len(local))) // 20 + 2)))
    ts = local["timestamp"]
    edges = pd.date_range(start=ts.iloc[0], end=ts.iloc[-1], periods=seg_count + 1)
    pnl_seg: list[float] = []
    win_seg: list[float] = []
    dd_seg: list[float] = []
    for i in range(seg_count):
        s, e = edges[i], edges[i + 1]
        m = (trades["entry_time"] >= s) & (trades["entry_time"] < e if i < seg_count - 1 else trades["entry_time"] <= e)
        seg = trades.loc[m]
        if seg.empty:
            pnl_seg.append(0.0)
            win_seg.append(0.0)
            dd_seg.append(0.0)
            continue
        net = pd.to_numeric(seg["net_pnl"], errors="coerce").fillna(0.0)
        cum = net.cumsum()
        run_max = cum.cummax()
        dd = ((cum - run_max).min()) if len(cum) else 0.0
        pnl_seg.append(float(net.sum()))
        win_seg.append(float((net > 0).mean() * 100.0))
        dd_seg.append(abs(float(dd)))

    pnl_consistency = _consistency_score(pnl_seg)
    win_consistency = _consistency_score(win_seg)
    dd_consistency = _consistency_score(dd_seg)
    time_stability = float((pnl_consistency + win_consistency + dd_consistency) / 3.0)

    half = max(1, seg_count // 2)
    early = float(np.mean(pnl_seg[:half])) if pnl_seg else 0.0
    late = float(np.mean(pnl_seg[half:])) if pnl_seg else 0.0
    decay_raw = max(0.0, early - late)
    decay_scale = float(np.std(pnl_seg, ddof=0) + abs(np.mean(pnl_seg)) + 1e-9)
    decay_score = float(decay_raw / (decay_raw + decay_scale))
    decay_flag = bool(late < early and decay_score > (1.0 - time_stability))

    adaptive_gap = return_scale / np.sqrt(max(1.0, float(len(tagged))))
    if tr - rg > adaptive_gap:
        base = "Performs best in trending markets"
    elif rg - tr > adaptive_gap:
        base = "Performs best in ranging markets"
    else:
        base = "Balanced between trending and ranging periods"

    if hv - lv > adaptive_gap:
        vol_note = "Sensitive to high volatility"
    elif lv - hv > adaptive_gap:
        vol_note = "Stable in low-volatility conditions"
    else:
        vol_note = "Volatility sensitivity is moderate"

    return {
        "high_vol_avg_return": round(hv, 4),
        "low_vol_avg_return": round(lv, 4),
        "trending_avg_return": round(tr, 4),
        "ranging_avg_return": round(rg, 4),
        "trend_confidence": round(trend_conf, 4),
        "volatility_confidence": round(vol_conf, 4),
        "context_confidence": round(context_conf, 4),
        "time_stability": round(time_stability, 4),
        "decay_score": round(decay_score, 4),
        "decay_flag": decay_flag,
        "sample_count": int(len(tagged)),
        "return_scale": round(return_scale, 6),
        "performance_context": f"{base}; {vol_note}",
    }


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
    perf_context = _performance_context_analysis(test_df, test_result.trades)

    return {
        "template": template,
        "params": merged_params,
        "full": full_result,
        "train": train_result,
        "test": test_result,
        "robustness_score": robustness,
        "performance_context": perf_context,
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
    elif template_key == "adaptive_indicator_mesh":
        for ema_fast in [8, 12, 20]:
            for ema_slow in [34, 55, 89]:
                if ema_fast >= ema_slow:
                    continue
                for rsi_len in [10, 14]:
                    for adx_min in [14, 18, 24]:
                        for vol_spike_mult in [1.0, 1.15, 1.35]:
                            for feature_pack in [
                                {"use_vwap": 1, "use_bbands": 1, "use_stoch": 1, "use_cci": 1, "use_williams": 1, "use_cmf": 0, "use_obv": 1},
                                {"use_vwap": 1, "use_bbands": 0, "use_stoch": 1, "use_cci": 1, "use_williams": 0, "use_cmf": 1, "use_obv": 1},
                                {"use_vwap": 0, "use_bbands": 1, "use_stoch": 1, "use_cci": 0, "use_williams": 1, "use_cmf": 1, "use_obv": 1},
                            ]:
                                variants.append(
                                    {
                                        "ema_fast": ema_fast,
                                        "ema_slow": ema_slow,
                                        "rsi_len": rsi_len,
                                        "rsi_long_min": 54,
                                        "rsi_short_max": 46,
                                        "adx_min": adx_min,
                                        "vol_spike_mult": vol_spike_mult,
                                        "bb_len": 20,
                                        "stoch_len": 14,
                                        "cci_len": 20,
                                        "williams_len": 14,
                                        "cmf_len": 20,
                                        **feature_pack,
                                    }
                                )
    else:
        variants.append(base_params)
    return variants or [base_params]


def _mutate_param_variants(template_key: str, params: dict[str, Any], rng: np.random.Generator, n: int = 12) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for _ in range(max(1, n)):
        p = dict(params)
        if template_key == "ema_cross_20_50":
            fast = int(max(3, min(60, p.get("ema_fast", 20) + int(rng.integers(-4, 5)))))
            slow = int(max(fast + 2, min(180, p.get("ema_slow", 50) + int(rng.integers(-8, 9)))))
            p.update({"ema_fast": fast, "ema_slow": slow})
        elif template_key == "rsi_reversal_30_70":
            rsi_len = int(max(5, min(35, p.get("rsi_len", 14) + int(rng.integers(-3, 4)))))
            oversold = float(max(10, min(45, p.get("oversold", 30) + int(rng.integers(-4, 5)))))
            overbought = float(max(55, min(90, p.get("overbought", 70) + int(rng.integers(-4, 5)))))
            if oversold >= overbought - 5:
                overbought = oversold + 6
            p.update({"rsi_len": rsi_len, "oversold": oversold, "overbought": overbought})
        elif template_key == "breakout_20":
            lookback = int(max(5, min(120, p.get("lookback", 20) + int(rng.integers(-8, 9)))))
            p.update({"lookback": lookback})
        elif template_key == "vwap_reclaim":
            ema_len = int(max(5, min(120, p.get("ema_len", 34) + int(rng.integers(-8, 9)))))
            vsm = float(max(0.8, min(3.0, p.get("vol_spike_mult", 1.5) + float(rng.normal(0, 0.18)))))
            p.update({"ema_len": ema_len, "vol_spike_mult": round(vsm, 3)})
        elif template_key == "multi_factor_combo":
            ema_fast = int(max(4, min(50, p.get("ema_fast", 20) + int(rng.integers(-5, 6)))))
            ema_slow = int(max(ema_fast + 3, min(220, p.get("ema_slow", 50) + int(rng.integers(-12, 13)))))
            rsi_len = int(max(5, min(34, p.get("rsi_len", 14) + int(rng.integers(-3, 4)))))
            rsi_long_min = float(max(45, min(68, p.get("rsi_long_min", 52) + int(rng.integers(-4, 5)))))
            rsi_short_max = float(max(32, min(55, p.get("rsi_short_max", 48) + int(rng.integers(-4, 5)))))
            if rsi_short_max >= rsi_long_min:
                rsi_short_max = rsi_long_min - 2.0
            adx_min = float(max(8, min(42, p.get("adx_min", 18) + int(rng.integers(-4, 5)))))
            vsm = float(max(0.8, min(3.0, p.get("vol_spike_mult", 1.2) + float(rng.normal(0, 0.15)))))
            p.update(
                {
                    "ema_fast": ema_fast,
                    "ema_slow": ema_slow,
                    "rsi_len": rsi_len,
                    "rsi_long_min": round(rsi_long_min, 2),
                    "rsi_short_max": round(rsi_short_max, 2),
                    "adx_min": round(adx_min, 2),
                    "vol_spike_mult": round(vsm, 3),
                }
            )
        elif template_key == "adaptive_indicator_mesh":
            ema_fast = int(max(4, min(40, p.get("ema_fast", 12) + int(rng.integers(-4, 5)))))
            ema_slow = int(max(ema_fast + 3, min(220, p.get("ema_slow", 55) + int(rng.integers(-10, 11)))))
            p.update(
                {
                    "ema_fast": ema_fast,
                    "ema_slow": ema_slow,
                    "rsi_len": int(max(6, min(30, p.get("rsi_len", 14) + int(rng.integers(-3, 4))))),
                    "rsi_long_min": float(max(48, min(70, p.get("rsi_long_min", 54) + int(rng.integers(-4, 5))))),
                    "rsi_short_max": float(max(30, min(52, p.get("rsi_short_max", 46) + int(rng.integers(-4, 5))))),
                    "adx_min": float(max(10, min(40, p.get("adx_min", 18) + int(rng.integers(-4, 5))))),
                    "vol_spike_mult": round(float(max(0.8, min(2.8, p.get("vol_spike_mult", 1.15) + float(rng.normal(0, 0.14))))), 3),
                    "use_vwap": int(rng.integers(0, 2)),
                    "use_bbands": int(rng.integers(0, 2)),
                    "use_stoch": int(rng.integers(0, 2)),
                    "use_cci": int(rng.integers(0, 2)),
                    "use_williams": int(rng.integers(0, 2)),
                    "use_cmf": int(rng.integers(0, 2)),
                    "use_obv": int(rng.integers(0, 2)),
                }
            )
            if p["use_vwap"] + p["use_bbands"] + p["use_stoch"] + p["use_cci"] + p["use_williams"] + p["use_cmf"] + p["use_obv"] < 2:
                p["use_vwap"] = 1
                p["use_obv"] = 1
        out.append(p)
    return out


def _param_signature(template_key: str, params: dict[str, Any]) -> str:
    bits = [template_key]
    for k in sorted(params.keys()):
        v = params[k]
        if isinstance(v, float):
            v = round(v, 3)
        bits.append(f"{k}={v}")
    return "|".join(bits)


def evolve_templates(
    df: pd.DataFrame,
    config: BacktestConfig | None = None,
    top_k: int = 8,
    progress_cb=None,
    result_cb=None,
    seed_pool: list[dict[str, Any]] | None = None,
    max_variants: int = 500,
    exploration_strength: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or BacktestConfig()
    all_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(42 + len(df) + top_k)

    grids = [{"template": t, "params": p, "origin": "random", "mutation_type": "base", "parent_id": "none"} for t in TEMPLATES for p in _variant_param_grid(t.key, t.params)]
    if seed_pool:
        template_map = {t.key: t for t in TEMPLATES}
        for seed in seed_pool:
            key = str(seed.get("template_key", ""))
            params = dict(seed.get("params", {}))
            t = template_map.get(key)
            if t is None:
                continue
            pid = str(seed.get("strategy_id", "seed"))
            grids.append({"template": t, "params": params, "origin": "mutation", "mutation_type": "elite_seed", "parent_id": pid})
            for tier, n_mut in [("minor", 8), ("medium", 8), ("major", 4)]:
                for mp in _mutate_param_variants(key, params, rng=rng, n=n_mut + int(8 * exploration_strength)):
                    mt = tier
                    tt = t
                    if tier == "major" and rng.random() < 0.35:
                        tt = TEMPLATES[int(rng.integers(0, len(TEMPLATES)))]
                        mp = dict(_variant_param_grid(tt.key, tt.params)[int(rng.integers(0, max(1, len(_variant_param_grid(tt.key, tt.params)))) )])
                        mt = "major_logic_swap"
                    grids.append({"template": tt, "params": mp, "origin": "mutation", "mutation_type": mt, "parent_id": pid})
        if len(seed_pool) >= 2:
            for _ in range(24):
                pa = seed_pool[int(rng.integers(0, len(seed_pool)))]
                pb = seed_pool[int(rng.integers(0, len(seed_pool)))]
                ta = template_map.get(str(pa.get("template_key", "")), TEMPLATES[0])
                source_a = dict(pa.get("params", {}))
                source_b = dict(pb.get("params", {}))
                child = {}
                keys = sorted(set(source_a.keys()) | set(source_b.keys()))
                for k in keys:
                    child[k] = source_a.get(k, source_b.get(k)) if rng.random() < 0.5 else source_b.get(k, source_a.get(k))
                grids.append(
                    {
                        "template": ta,
                        "params": child or dict(ta.params),
                        "origin": "crossover",
                        "mutation_type": "recombine",
                        "parent_id": f"{pa.get('strategy_id', 'A')}+{pb.get('strategy_id', 'B')}",
                    }
                )

    if exploration_strength > 0:
        extra = int(max_variants * min(0.7, 0.25 + exploration_strength))
        for _ in range(extra):
            tt = TEMPLATES[int(rng.integers(0, len(TEMPLATES)))]
            pv = _variant_param_grid(tt.key, tt.params)
            pp = dict(pv[int(rng.integers(0, max(1, len(pv))))])
            grids.append({"template": tt, "params": pp, "origin": "random", "mutation_type": "explore_inject", "parent_id": "none"})

    dedup: dict[str, dict[str, Any]] = {}
    for g in grids:
        key = _param_signature(g["template"].key, g["params"])
        dedup[key] = g
    grids = list(dedup.values())
    if max_variants > 0 and len(grids) > max_variants:
        idxs = np.linspace(0, len(grids) - 1, num=max_variants, dtype=int).tolist()
        grids = [grids[i] for i in idxs]

    total = max(1, len(grids))

    for idx, g in enumerate(grids, start=1):
        t = g["template"]
        params = g["params"]
        if progress_cb is not None and (idx == 1 or idx % 3 == 0 or idx == total):
            try:
                progress_cb(idx, total, t.name)
            except Exception:
                pass
        try:
            evaluated = evaluate_template(df, t.key, params=params, config=cfg)
            test = evaluated["test"].metrics
            perf_context = dict(evaluated.get("performance_context", {}))
            test_trades = evaluated["test"].trades
            avg_trade_return_pct = float(test_trades["return_pct"].mean()) if not test_trades.empty else 0.0
            max_win_pct = float(test_trades["return_pct"].max()) if not test_trades.empty else 0.0
            max_loss_pct = float(test_trades["return_pct"].min()) if not test_trades.empty else 0.0
            wins = int((test_trades["net_pnl"] > 0).sum()) if not test_trades.empty else 0
            losses = int((test_trades["net_pnl"] < 0).sum()) if not test_trades.empty else 0
            trade_returns = pd.to_numeric(test_trades["return_pct"], errors="coerce").dropna() if not test_trades.empty else pd.Series(dtype=float)
            n_trades = max(1, int(test["total_trades"]))
            trade_stability = float(1.0 - np.exp(-np.sqrt(float(n_trades))))
            ret_mean = float(trade_returns.mean()) if not trade_returns.empty else 0.0
            ret_std = float(trade_returns.std(ddof=0)) if not trade_returns.empty else 0.0
            return_consistency = float(1.0 / (1.0 + (ret_std / (abs(ret_mean) + 1e-9))))
            dd_scale = abs(ret_mean) + ret_std + 1e-9
            dd_stability = float(1.0 / (1.0 + abs(float(test["max_drawdown_pct"])) / (dd_scale * 100.0 + 1e-9)))
            time_stability = float(perf_context.get("time_stability", 0.0))
            behavior_robustness = round((trade_stability + return_consistency + dd_stability + time_stability) / 4.0 * 100.0, 2)
            complexity_score = float(len(t.indicators)) * 1.8 + float(len(params)) * 0.35
            row = {
                "strategy": t.name,
                "template_key": t.key,
                "params": evaluated["params"],
                "origin": g.get("origin", "random"),
                "mutation_type": g.get("mutation_type", "base"),
                "parent_id": g.get("parent_id", "none"),
                "complexity_score": complexity_score,
                "robustness_score": float(evaluated["robustness_score"]),
                "test_return_pct": float(test["total_return_pct"]),
                "test_win_rate_pct": float(test["win_rate_pct"]),
                "test_max_drawdown_pct": float(test["max_drawdown_pct"]),
                "test_trades": int(test["total_trades"]),
                "test_avg_trade_return_pct": avg_trade_return_pct,
                "test_max_win_pct": max_win_pct,
                "test_max_loss_pct": max_loss_pct,
                "test_win_trades": wins,
                "test_loss_trades": losses,
                "ctx_high_vol_avg_return": float(perf_context.get("high_vol_avg_return", 0.0)),
                "ctx_low_vol_avg_return": float(perf_context.get("low_vol_avg_return", 0.0)),
                "ctx_trending_avg_return": float(perf_context.get("trending_avg_return", 0.0)),
                "ctx_ranging_avg_return": float(perf_context.get("ranging_avg_return", 0.0)),
                "ctx_trend_confidence": float(perf_context.get("trend_confidence", 0.0)),
                "ctx_volatility_confidence": float(perf_context.get("volatility_confidence", 0.0)),
                "ctx_confidence": float(perf_context.get("context_confidence", 0.0)),
                "ctx_time_stability": float(perf_context.get("time_stability", 0.0)),
                "ctx_decay_score": float(perf_context.get("decay_score", 0.0)),
                "ctx_decay_flag": bool(perf_context.get("decay_flag", False)),
                "ctx_sample_count": int(perf_context.get("sample_count", 0)),
                "ctx_return_scale": float(perf_context.get("return_scale", 0.0)),
                "performance_context": str(perf_context.get("performance_context", "")),
                "behavior_robustness": behavior_robustness,
            }
            all_rows.append(row)
            if result_cb is not None:
                try:
                    result_cb(idx, total, dict(row))
                except Exception:
                    pass
        except Exception:
            continue

    if not all_rows:
        raise ValueError("Evolution engine could not evaluate any template variants")

    frame = pd.DataFrame(all_rows)
    frame["structure_sig"] = frame.apply(lambda r: _param_signature(str(r["template_key"]), dict(r["params"])), axis=1)
    dup_counts = frame["structure_sig"].value_counts()
    fam_counts = frame["template_key"].value_counts()
    frame["dup_penalty"] = frame["structure_sig"].map(lambda s: max(0.0, float(dup_counts.get(s, 1) - 1) * 1.6))
    frame["family_penalty"] = frame["template_key"].map(lambda k: max(0.0, float(fam_counts.get(k, 1) / max(1, len(frame)) * 10.0)))
    frame["fitness"] = (
        frame["robustness_score"] * 0.50 +
        frame["test_return_pct"] * 0.30 +
        frame["test_win_rate_pct"] * 0.10 -
        frame["test_max_drawdown_pct"].abs() * 0.10 +
        frame["complexity_score"] * 0.05 -
        frame["dup_penalty"] -
        frame["family_penalty"]
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
