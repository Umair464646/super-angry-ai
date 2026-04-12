from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REGIMES = [
    "trending",
    "ranging",
    "breakout_phase",
    "compression",
    "high_volatility",
    "low_activity",
]


@dataclass
class AIAnalysisResult:
    summary: dict
    regime_counts: dict[str, int]
    confidence_distribution: dict[str, int]
    top_setups: pd.DataFrame
    loss_curve: list[float]
    accuracy_curve: list[float]


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"AI analysis requires missing columns: {', '.join(missing)}")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    local = df.copy().sort_values("timestamp").reset_index(drop=True)

    close = local["close"].astype(float)
    high = local["high"].astype(float)
    low = local["low"].astype(float)
    volume = local["volume"].astype(float)

    returns = close.pct_change().fillna(0.0)
    log_returns = np.log(close.replace(0, np.nan)).diff().fillna(0.0)

    local["ret_1"] = returns
    local["log_ret_1"] = log_returns
    local["vol_30"] = returns.rolling(30).std().fillna(0.0)
    local["vol_120"] = returns.rolling(120).std().fillna(0.0)

    ema_fast = close.ewm(span=20, adjust=False).mean()
    ema_slow = close.ewm(span=80, adjust=False).mean()
    local["trend_strength"] = ((ema_fast - ema_slow) / close.replace(0, np.nan)).fillna(0.0)

    rolling_range = (high.rolling(40).max() - low.rolling(40).min()).replace(0, np.nan)
    local["compression_ratio"] = ((high - low) / rolling_range).fillna(0.0)

    local["range_breakout"] = (
        (close > high.rolling(20).max().shift(1)) |
        (close < low.rolling(20).min().shift(1))
    ).fillna(False)

    local["volume_z"] = (
        (volume - volume.rolling(120).mean()) /
        volume.rolling(120).std().replace(0, np.nan)
    ).fillna(0.0)

    if "synthetic" in local.columns:
        local["synthetic"] = local["synthetic"].fillna(0).astype(int)
    else:
        local["synthetic"] = 0

    return local


def _classify_regimes(df: pd.DataFrame) -> pd.Series:
    trend = df["trend_strength"].abs()
    vol = df["vol_30"]
    vol_base = df["vol_120"].replace(0, np.nan).fillna(df["vol_30"].mean() + 1e-12)
    vol_ratio = (vol / vol_base).fillna(0.0)
    compression = df["compression_ratio"]
    breakout = df["range_breakout"].astype(bool)
    low_activity = (df["volume_z"] < -0.8) | (df["synthetic"] == 1)

    regime = pd.Series("ranging", index=df.index)
    regime[(trend > 0.0015) & (vol_ratio < 1.8)] = "trending"
    regime[(compression < 0.16) & (vol_ratio < 1.2)] = "compression"
    regime[(vol_ratio > 2.0)] = "high_volatility"
    regime[breakout] = "breakout_phase"
    regime[low_activity] = "low_activity"

    return regime


def _train_setup_model(df: pd.DataFrame, epochs: int = 24, lr: float = 0.08):
    feature_cols = [
        "ret_1",
        "log_ret_1",
        "vol_30",
        "trend_strength",
        "compression_ratio",
        "volume_z",
        "synthetic",
    ]

    X = df[feature_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y = (df["close"].shift(-1) > df["close"]).astype(float).fillna(0.0).to_numpy()

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xn = (X - mean) / std

    n, m = Xn.shape
    w = np.zeros(m)
    b = 0.0
    losses: list[float] = []
    accuracies: list[float] = []

    for _ in range(epochs):
        logits = Xn @ w + b
        p = _sigmoid(logits)

        eps = 1e-9
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        preds = (p >= 0.5).astype(float)
        acc = float((preds == y).mean())

        grad_w = (Xn.T @ (p - y)) / n
        grad_b = float(np.mean(p - y))

        w -= lr * grad_w
        b -= lr * grad_b

        losses.append(float(loss))
        accuracies.append(acc)

    probs = _sigmoid(Xn @ w + b)
    confidence = np.clip(np.abs(probs - 0.5) * 2.0, 0.0, 1.0)

    return probs, confidence, losses, accuracies


def analyze_market_ai(df: pd.DataFrame) -> AIAnalysisResult:
    _require_columns(df, ["timestamp", "open", "high", "low", "close", "volume"])

    local = _build_features(df)
    local["regime"] = _classify_regimes(local)
    probs, confidence, losses, accuracies = _train_setup_model(local)

    local["setup_probability"] = probs
    local["setup_confidence"] = confidence
    local["direction"] = np.where(local["setup_probability"] >= 0.5, "long_bias", "short_bias")

    regime_counts = (
        local["regime"].value_counts().reindex(REGIMES, fill_value=0).astype(int).to_dict()
    )

    confidence_bins = pd.cut(
        local["setup_confidence"],
        bins=[-0.001, 0.25, 0.5, 0.75, 1.0],
        labels=["low", "moderate", "high", "very_high"],
    )
    confidence_distribution = (
        confidence_bins.value_counts().reindex(["low", "moderate", "high", "very_high"], fill_value=0).astype(int).to_dict()
    )

    top = local[[
        "timestamp",
        "close",
        "regime",
        "direction",
        "setup_probability",
        "setup_confidence",
        "synthetic",
    ]].copy()

    top = top[(top["synthetic"] == 0) & (top["setup_confidence"] >= 0.70)]
    top = top.sort_values("setup_confidence", ascending=False).head(200).reset_index(drop=True)

    summary = {
        "rows": int(len(local)),
        "avg_confidence": float(local["setup_confidence"].mean()),
        "high_confidence_rows": int((local["setup_confidence"] >= 0.70).sum()),
        "train_final_loss": float(losses[-1]),
        "train_final_accuracy": float(accuracies[-1]),
    }

    return AIAnalysisResult(
        summary=summary,
        regime_counts=regime_counts,
        confidence_distribution=confidence_distribution,
        top_setups=top,
        loss_curve=losses,
        accuracy_curve=accuracies,
    )
