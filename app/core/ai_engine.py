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
    prediction_distribution: dict[str, int]
    top_setups: pd.DataFrame
    loss_curve: list[float]
    accuracy_curve: list[float]
    val_loss_curve: list[float]
    val_accuracy_curve: list[float]
    precision_curve: list[float]
    recall_curve: list[float]
    f1_curve: list[float]
    lr_curve: list[float]
    grad_norm_curve: list[float]
    drift_curve: list[float]
    feature_importance: dict[str, float]
    model_notes: str
    nn_architecture: str


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


def _train_setup_model(
    df: pd.DataFrame,
    epochs: int = 24,
    lr: float = 0.08,
    model_type: str = "mlp",
    epoch_cb=None,
):
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
    split = max(32, int(n * 0.8))
    split = min(split, n - 8)
    Xtr, Xva = Xn[:split], Xn[split:]
    ytr, yva = y[:split], y[split:]
    if len(Xva) == 0:
        Xtr, Xva = Xn, Xn
        ytr, yva = y, y

    losses: list[float] = []
    accuracies: list[float] = []
    val_losses: list[float] = []
    val_accuracies: list[float] = []
    precision_curve: list[float] = []
    recall_curve: list[float] = []
    f1_curve: list[float] = []
    lr_curve: list[float] = []
    grad_norm_curve: list[float] = []
    drift_curve: list[float] = []

    if model_type == "logistic":
        w = np.zeros(m)
        b = 0.0
        for epoch in range(epochs):
            lr_e = lr * (0.985 ** epoch)
            logits = Xtr @ w + b
            p = _sigmoid(logits)

            eps = 1e-9
            loss = -np.mean(ytr * np.log(p + eps) + (1 - ytr) * np.log(1 - p + eps))
            preds = (p >= 0.5).astype(float)
            acc = float((preds == ytr).mean())

            grad_w = (Xtr.T @ (p - ytr)) / max(1, len(Xtr))
            grad_b = float(np.mean(p - ytr))
            grad_norm = float(np.sqrt(np.sum(grad_w ** 2) + grad_b**2))

            w -= lr_e * grad_w
            b -= lr_e * grad_b
            losses.append(float(loss))
            accuracies.append(acc)
            pva = _sigmoid(Xva @ w + b)
            eps = 1e-9
            vloss = -np.mean(yva * np.log(pva + eps) + (1 - yva) * np.log(1 - pva + eps))
            vpred = (pva >= 0.5).astype(float)
            vacc = float((vpred == yva).mean())
            tp = float(((vpred == 1) & (yva == 1)).sum())
            fp = float(((vpred == 1) & (yva == 0)).sum())
            fn = float(((vpred == 0) & (yva == 1)).sum())
            precision = tp / max(1.0, tp + fp)
            recall = tp / max(1.0, tp + fn)
            f1 = (2 * precision * recall) / max(1e-9, precision + recall)
            drift = abs(float(vloss) - float(loss))
            val_losses.append(float(vloss))
            val_accuracies.append(vacc)
            precision_curve.append(float(precision))
            recall_curve.append(float(recall))
            f1_curve.append(float(f1))
            lr_curve.append(float(lr_e))
            grad_norm_curve.append(float(grad_norm))
            drift_curve.append(float(drift))
            if epoch_cb is not None:
                epoch_cb(epoch + 1, epochs, float(loss), acc, {
                    "val_loss": float(vloss),
                    "val_acc": vacc,
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "lr": float(lr_e),
                    "grad_norm": float(grad_norm),
                    "drift": float(drift),
                    "layer_activity": [float(np.abs(w).mean()), float(abs(b))],
                    "feature_strength": np.abs(w).astype(float).tolist(),
                    "output_confidence": float(np.mean(np.abs(pva - 0.5) * 2.0)),
                })

        probs = _sigmoid(Xn @ w + b)
        arch = f"Input({m}) -> Logistic(1)"
        feature_importance = {feature_cols[i]: float(abs(w[i])) for i in range(m)}
    else:
        hidden = max(8, min(24, m * 2))
        rng = np.random.default_rng(42)
        w1 = rng.normal(0, 0.1, size=(m, hidden))
        b1 = np.zeros(hidden)
        w2 = rng.normal(0, 0.1, size=(hidden, 1))
        b2 = np.zeros(1)
        for epoch in range(epochs):
            lr_e = lr * (0.985 ** epoch)
            z1 = Xtr @ w1 + b1
            a1 = np.tanh(z1)
            z2 = a1 @ w2 + b2
            p = _sigmoid(z2.reshape(-1))

            eps = 1e-9
            loss = -np.mean(ytr * np.log(p + eps) + (1 - ytr) * np.log(1 - p + eps))
            preds = (p >= 0.5).astype(float)
            acc = float((preds == ytr).mean())

            dz2 = (p - ytr).reshape(-1, 1) / max(1, len(Xtr))
            dw2 = a1.T @ dz2
            db2 = dz2.sum(axis=0)
            da1 = dz2 @ w2.T
            dz1 = da1 * (1 - np.tanh(z1) ** 2)
            dw1 = Xtr.T @ dz1
            db1 = dz1.sum(axis=0)
            grad_norm = float(np.sqrt(np.sum(dw1 ** 2) + np.sum(dw2 ** 2)))

            w2 -= lr_e * dw2
            b2 -= lr_e * db2
            w1 -= lr_e * dw1
            b1 -= lr_e * db1

            losses.append(float(loss))
            accuracies.append(acc)
            v1 = np.tanh(Xva @ w1 + b1)
            pva = _sigmoid((v1 @ w2 + b2).reshape(-1))
            eps = 1e-9
            vloss = -np.mean(yva * np.log(pva + eps) + (1 - yva) * np.log(1 - pva + eps))
            vpred = (pva >= 0.5).astype(float)
            vacc = float((vpred == yva).mean())
            tp = float(((vpred == 1) & (yva == 1)).sum())
            fp = float(((vpred == 1) & (yva == 0)).sum())
            fn = float(((vpred == 0) & (yva == 1)).sum())
            precision = tp / max(1.0, tp + fp)
            recall = tp / max(1.0, tp + fn)
            f1 = (2 * precision * recall) / max(1e-9, precision + recall)
            drift = abs(float(vloss) - float(loss))
            val_losses.append(float(vloss))
            val_accuracies.append(vacc)
            precision_curve.append(float(precision))
            recall_curve.append(float(recall))
            f1_curve.append(float(f1))
            lr_curve.append(float(lr_e))
            grad_norm_curve.append(float(grad_norm))
            drift_curve.append(float(drift))
            if epoch_cb is not None:
                epoch_cb(epoch + 1, epochs, float(loss), acc, {
                    "val_loss": float(vloss),
                    "val_acc": vacc,
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "lr": float(lr_e),
                    "grad_norm": float(grad_norm),
                    "drift": float(drift),
                    "layer_activity": [
                        float(np.abs(Xtr).mean()),
                        float(np.abs(a1).mean()),
                        float(np.abs(p).mean()),
                    ],
                    "feature_strength": np.abs(dw1).mean(axis=1).astype(float).tolist(),
                    "output_confidence": float(np.mean(np.abs(pva - 0.5) * 2.0)),
                })

        probs = _sigmoid((np.tanh(Xn @ w1 + b1) @ w2 + b2).reshape(-1))
        arch = f"Input({m}) -> Dense({hidden}, tanh) -> Dense(1, sigmoid)"
        feature_importance = {feature_cols[i]: float(np.abs(w1[i]).mean()) for i in range(m)}

    confidence = np.clip(np.abs(probs - 0.5) * 2.0, 0.0, 1.0)

    return probs, confidence, losses, accuracies, arch, {
        "val_loss_curve": val_losses,
        "val_accuracy_curve": val_accuracies,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
        "f1_curve": f1_curve,
        "lr_curve": lr_curve,
        "grad_norm_curve": grad_norm_curve,
        "drift_curve": drift_curve,
        "feature_importance": feature_importance,
        "feature_cols": feature_cols,
    }


def analyze_market_ai(df: pd.DataFrame, model_type: str = "mlp", epoch_cb=None) -> AIAnalysisResult:
    _require_columns(df, ["timestamp", "open", "high", "low", "close", "volume"])

    local = _build_features(df)
    local["regime"] = _classify_regimes(local)
    probs, confidence, losses, accuracies, arch, diag = _train_setup_model(
        local, model_type=model_type, epoch_cb=epoch_cb
    )

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
    pred_bins = pd.cut(
        local["setup_probability"],
        bins=[-0.001, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
    )
    prediction_distribution = (
        pred_bins.value_counts()
        .reindex(["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"], fill_value=0)
        .astype(int)
        .to_dict()
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
        "best_val_loss": float(min(diag["val_loss_curve"]) if diag["val_loss_curve"] else losses[-1]),
        "final_drift": float(diag["drift_curve"][-1] if diag["drift_curve"] else 0.0),
    }

    return AIAnalysisResult(
        summary=summary,
        regime_counts=regime_counts,
        confidence_distribution=confidence_distribution,
        prediction_distribution=prediction_distribution,
        top_setups=top,
        loss_curve=losses,
        accuracy_curve=accuracies,
        val_loss_curve=diag["val_loss_curve"],
        val_accuracy_curve=diag["val_accuracy_curve"],
        precision_curve=diag["precision_curve"],
        recall_curve=diag["recall_curve"],
        f1_curve=diag["f1_curve"],
        lr_curve=diag["lr_curve"],
        grad_norm_curve=diag["grad_norm_curve"],
        drift_curve=diag["drift_curve"],
        feature_importance=diag["feature_importance"],
        model_notes=(
            f"Setup model type: {model_type}.\\n"
            "Inputs: ret_1, log_ret_1, vol_30, trend_strength, compression_ratio, volume_z, synthetic.\\n"
            "Objective: predict next-bar direction probability with transparent confidence."
        ),
        nn_architecture=arch,
    )
