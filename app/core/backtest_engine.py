from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Literal

import numpy as np
import pandas as pd


Side = Literal["long", "short"]


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    fee_rate: float = 0.0004          # 0.04%
    slippage_rate: float = 0.0002     # 0.02%
    risk_pct_per_trade: float = 0.10  # 10% of current equity used as position value
    allow_long: bool = True
    allow_short: bool = True
    stop_loss_pct: float = 0.01       # 1%
    take_profit_pct: float = 0.02     # 2%
    one_position_at_a_time: bool = True


@dataclass
class TradeRecord:
    entry_time: str
    exit_time: str
    side: str
    entry_price: float
    exit_price: float
    qty: float
    gross_pnl: float
    fees: float
    net_pnl: float
    return_pct: float
    exit_reason: str
    bars_held: int


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict


def _validate_backtest_input(df: pd.DataFrame) -> None:
    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "long_entry",
        "short_entry",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Backtest input missing required columns: {', '.join(missing)}")

    if df.empty:
        raise ValueError("Cannot backtest an empty dataframe")


def _normalize_signal_column(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    return numeric.astype(int) != 0


def _entry_fill_price(next_open: float, side: Side, slippage_rate: float) -> float:
    if side == "long":
        return float(next_open) * (1.0 + slippage_rate)
    return float(next_open) * (1.0 - slippage_rate)


def _exit_fill_price(raw_price: float, side: Side, slippage_rate: float) -> float:
    if side == "long":
        return float(raw_price) * (1.0 - slippage_rate)
    return float(raw_price) * (1.0 + slippage_rate)


def _compute_position_qty(equity: float, entry_price: float, risk_pct_per_trade: float) -> float:
    position_value = equity * risk_pct_per_trade
    if entry_price <= 0:
        return 0.0
    return position_value / entry_price


def _long_stop_price(entry_price: float, stop_loss_pct: float) -> float:
    return entry_price * (1.0 - stop_loss_pct)


def _long_take_price(entry_price: float, take_profit_pct: float) -> float:
    return entry_price * (1.0 + take_profit_pct)


def _short_stop_price(entry_price: float, stop_loss_pct: float) -> float:
    return entry_price * (1.0 + stop_loss_pct)


def _short_take_price(entry_price: float, take_profit_pct: float) -> float:
    return entry_price * (1.0 - take_profit_pct)


def _trade_fees(entry_price: float, exit_price: float, qty: float, fee_rate: float) -> float:
    entry_notional = abs(entry_price * qty)
    exit_notional = abs(exit_price * qty)
    return (entry_notional + exit_notional) * fee_rate


def _build_metrics(
    initial_capital: float,
    final_equity: float,
    equity_curve: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> dict:
    total_return_pct = ((final_equity / initial_capital) - 1.0) * 100.0

    if trades_df.empty:
        return {
            "initial_capital": initial_capital,
            "final_equity": final_equity,
            "total_return_pct": total_return_pct,
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
            "avg_trade_net_pnl": 0.0,
            "max_drawdown_pct": 0.0,
        }

    wins = trades_df["net_pnl"] > 0
    losses = trades_df["net_pnl"] < 0

    gross_profit = float(trades_df.loc[wins, "net_pnl"].sum()) if wins.any() else 0.0
    gross_loss = float(-trades_df.loc[losses, "net_pnl"].sum()) if losses.any() else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    eq = equity_curve["equity"].astype(float)
    running_max = eq.cummax()
    drawdown = (eq / running_max) - 1.0
    max_drawdown_pct = float(drawdown.min() * 100.0) if len(drawdown) else 0.0

    return {
        "initial_capital": float(initial_capital),
        "final_equity": float(final_equity),
        "total_return_pct": float(total_return_pct),
        "total_trades": int(len(trades_df)),
        "win_rate_pct": float(wins.mean() * 100.0),
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
        "profit_factor": float(profit_factor if np.isfinite(profit_factor) else 999999.0),
        "avg_trade_net_pnl": float(trades_df["net_pnl"].mean()),
        "max_drawdown_pct": float(max_drawdown_pct),
    }


def run_backtest(df: pd.DataFrame, config: Optional[BacktestConfig] = None) -> BacktestResult:
    config = config or BacktestConfig()

    local = df.copy()
    _validate_backtest_input(local)

    local["timestamp"] = pd.to_datetime(local["timestamp"], utc=True, errors="coerce")
    local = local.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    local["long_entry"] = _normalize_signal_column(local["long_entry"])
    local["short_entry"] = _normalize_signal_column(local["short_entry"])

    equity = float(config.initial_capital)
    equity_points: list[dict] = []
    trades: list[TradeRecord] = []

    in_position = False
    position_side: Optional[Side] = None
    entry_time = None
    entry_price = 0.0
    entry_index = -1
    qty = 0.0
    stop_price = 0.0
    take_price = 0.0

    for i in range(len(local)):
        row = local.iloc[i]
        ts = row["timestamp"]

        equity_points.append({
            "timestamp": ts,
            "equity": equity,
        })

        if i >= len(local) - 1:
            continue

        next_row = local.iloc[i + 1]

        if not in_position:
            can_long = bool(row["long_entry"]) and config.allow_long
            can_short = bool(row["short_entry"]) and config.allow_short

            if can_long and can_short:
                # if both happen, skip ambiguous bar
                continue

            if can_long:
                fill = _entry_fill_price(next_row["open"], "long", config.slippage_rate)
                trade_qty = _compute_position_qty(equity, fill, config.risk_pct_per_trade)
                if trade_qty > 0:
                    in_position = True
                    position_side = "long"
                    entry_time = next_row["timestamp"]
                    entry_price = fill
                    entry_index = i + 1
                    qty = trade_qty
                    stop_price = _long_stop_price(entry_price, config.stop_loss_pct)
                    take_price = _long_take_price(entry_price, config.take_profit_pct)

            elif can_short:
                fill = _entry_fill_price(next_row["open"], "short", config.slippage_rate)
                trade_qty = _compute_position_qty(equity, fill, config.risk_pct_per_trade)
                if trade_qty > 0:
                    in_position = True
                    position_side = "short"
                    entry_time = next_row["timestamp"]
                    entry_price = fill
                    entry_index = i + 1
                    qty = trade_qty
                    stop_price = _short_stop_price(entry_price, config.stop_loss_pct)
                    take_price = _short_take_price(entry_price, config.take_profit_pct)

            continue

        # manage open trade
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        exit_reason = None
        raw_exit_price = None

        if position_side == "long":
            stop_hit = low <= stop_price
            take_hit = high >= take_price

            if stop_hit and take_hit:
                # pessimistic assumption: stop first
                exit_reason = "stop_loss"
                raw_exit_price = stop_price
            elif stop_hit:
                exit_reason = "stop_loss"
                raw_exit_price = stop_price
            elif take_hit:
                exit_reason = "take_profit"
                raw_exit_price = take_price

        elif position_side == "short":
            stop_hit = high >= stop_price
            take_hit = low <= take_price

            if stop_hit and take_hit:
                exit_reason = "stop_loss"
                raw_exit_price = stop_price
            elif stop_hit:
                exit_reason = "stop_loss"
                raw_exit_price = stop_price
            elif take_hit:
                exit_reason = "take_profit"
                raw_exit_price = take_price

        if exit_reason is None and i == len(local) - 2:
            exit_reason = "end_of_data"
            raw_exit_price = close

        if exit_reason is None:
            continue

        exit_price = _exit_fill_price(raw_exit_price, position_side, config.slippage_rate)

        if position_side == "long":
            gross_pnl = (exit_price - entry_price) * qty
        else:
            gross_pnl = (entry_price - exit_price) * qty

        fees = _trade_fees(entry_price, exit_price, qty, config.fee_rate)
        net_pnl = gross_pnl - fees
        equity += net_pnl

        invested_value = max(entry_price * qty, 1e-12)
        return_pct = (net_pnl / invested_value) * 100.0
        bars_held = i - entry_index + 1

        trades.append(
            TradeRecord(
                entry_time=str(entry_time),
                exit_time=str(ts),
                side=position_side,
                entry_price=float(entry_price),
                exit_price=float(exit_price),
                qty=float(qty),
                gross_pnl=float(gross_pnl),
                fees=float(fees),
                net_pnl=float(net_pnl),
                return_pct=float(return_pct),
                exit_reason=str(exit_reason),
                bars_held=int(bars_held),
            )
        )

        in_position = False
        position_side = None
        entry_time = None
        entry_price = 0.0
        entry_index = -1
        qty = 0.0
        stop_price = 0.0
        take_price = 0.0

        equity_points.append({
            "timestamp": ts,
            "equity": equity,
        })

    equity_curve = pd.DataFrame(equity_points).drop_duplicates(subset=["timestamp"], keep="last")
    trades_df = pd.DataFrame([asdict(t) for t in trades])

    metrics = _build_metrics(
        initial_capital=config.initial_capital,
        final_equity=equity,
        equity_curve=equity_curve,
        trades_df=trades_df,
    )

    return BacktestResult(
        equity_curve=equity_curve,
        trades=trades_df,
        metrics=metrics,
    )