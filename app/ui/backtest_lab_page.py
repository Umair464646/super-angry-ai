from __future__ import annotations

import pandas as pd

from PyQt6.QtCore import pyqtSignal, QThread
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QTextEdit,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
    QDoubleSpinBox,
    QCheckBox,
    QSplitter,
)

from app.core.backtest_engine import BacktestConfig
from app.core.backtest_worker import BacktestWorker


STRATEGY_PRESETS = [
    "EMA Cross 20/50",
    "RSI Reversal 30/70",
    "Breakout 20",
]


class BacktestLabPage(QWidget):
    log_message = pyqtSignal(str, str)
    timeframe_requested = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.source_path = None
        self.timeframe_cache = {}

        self.backtest_thread = None
        self.backtest_worker = None
        self.backtest_result = None

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Backtest Lab")
        title.setStyleSheet("font-size: 24px; font-weight: 700;")

        subtitle = QLabel(
            "Run simple rule-based backtests on cached timeframe datasets with next-bar execution, fees, slippage, stop loss, and take profit."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #8a95a5;")

        layout.addWidget(title)
        layout.addWidget(subtitle)

        controls_top = QHBoxLayout()

        self.timeframe_box = QComboBox()
        self.timeframe_box.addItems(["1s", "5s", "15s", "30s", "1m", "5m", "15m", "1h", "4h"])
        self.timeframe_box.setCurrentText("1m")
        self.timeframe_box.currentTextChanged.connect(self._ensure_timeframe_ready)

        self.strategy_box = QComboBox()
        self.strategy_box.addItems(STRATEGY_PRESETS)
        self.strategy_box.setCurrentText("EMA Cross 20/50")

        self.run_btn = QPushButton("Run Backtest")
        self.run_btn.clicked.connect(self.run_backtest)

        controls_top.addWidget(QLabel("Timeframe"))
        controls_top.addWidget(self.timeframe_box)
        controls_top.addWidget(QLabel("Strategy"))
        controls_top.addWidget(self.strategy_box)
        controls_top.addWidget(self.run_btn)
        controls_top.addStretch(1)

        layout.addLayout(controls_top)

        config_row = QHBoxLayout()

        self.initial_capital = QDoubleSpinBox()
        self.initial_capital.setRange(100.0, 100000000.0)
        self.initial_capital.setDecimals(2)
        self.initial_capital.setValue(10000.0)
        self.initial_capital.setPrefix("$")

        self.fee_rate = QDoubleSpinBox()
        self.fee_rate.setRange(0.0, 0.05)
        self.fee_rate.setDecimals(5)
        self.fee_rate.setSingleStep(0.0001)
        self.fee_rate.setValue(0.0004)

        self.slippage_rate = QDoubleSpinBox()
        self.slippage_rate.setRange(0.0, 0.05)
        self.slippage_rate.setDecimals(5)
        self.slippage_rate.setSingleStep(0.0001)
        self.slippage_rate.setValue(0.0002)

        self.risk_pct = QDoubleSpinBox()
        self.risk_pct.setRange(0.001, 1.0)
        self.risk_pct.setDecimals(3)
        self.risk_pct.setSingleStep(0.01)
        self.risk_pct.setValue(0.10)

        self.stop_loss_pct = QDoubleSpinBox()
        self.stop_loss_pct.setRange(0.001, 0.50)
        self.stop_loss_pct.setDecimals(3)
        self.stop_loss_pct.setSingleStep(0.001)
        self.stop_loss_pct.setValue(0.01)

        self.take_profit_pct = QDoubleSpinBox()
        self.take_profit_pct.setRange(0.001, 1.00)
        self.take_profit_pct.setDecimals(3)
        self.take_profit_pct.setSingleStep(0.001)
        self.take_profit_pct.setValue(0.02)

        self.allow_long = QCheckBox("Allow Long")
        self.allow_long.setChecked(True)

        self.allow_short = QCheckBox("Allow Short")
        self.allow_short.setChecked(True)

        config_row.addWidget(QLabel("Capital"))
        config_row.addWidget(self.initial_capital)
        config_row.addWidget(QLabel("Fee"))
        config_row.addWidget(self.fee_rate)
        config_row.addWidget(QLabel("Slippage"))
        config_row.addWidget(self.slippage_rate)
        config_row.addWidget(QLabel("Risk %"))
        config_row.addWidget(self.risk_pct)
        config_row.addWidget(QLabel("SL %"))
        config_row.addWidget(self.stop_loss_pct)
        config_row.addWidget(QLabel("TP %"))
        config_row.addWidget(self.take_profit_pct)
        config_row.addWidget(self.allow_long)
        config_row.addWidget(self.allow_short)
        config_row.addStretch(1)

        layout.addLayout(config_row)

        self.stage_label = QLabel("Stage: idle")
        self.stage_label.setStyleSheet("color: #8a95a5;")

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        layout.addWidget(self.stage_label)
        layout.addWidget(self.progress)

        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setMinimumHeight(140)

        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])

        self.trades_table = QTableWidget(0, 0)

        splitter = QSplitter()

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Metrics"))
        left_layout.addWidget(self.metrics_table, 1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QLabel("Trades Preview"))
        right_layout.addWidget(self.trades_table, 1)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([350, 1000])

        layout.addWidget(self.summary_box)
        layout.addWidget(splitter, 1)

        self._refresh_summary()

    def set_source_context(self, source_path: str, timeframe_cache: dict):
        self.source_path = source_path
        self.timeframe_cache = timeframe_cache
        self._refresh_summary()

    def set_timeframe_dataset(self, timeframe: str, df):
        self.timeframe_cache[timeframe] = df
        self._refresh_summary()

    def _ensure_timeframe_ready(self):
        tf = self.timeframe_box.currentText()
        if tf not in self.timeframe_cache:
            self.log_message.emit("INFO", f"Backtest Lab requested timeframe build: {tf}")
            self.timeframe_requested.emit(tf)
        self._refresh_summary()

    def _refresh_summary(self):
        tf = self.timeframe_box.currentText()
        df = self.timeframe_cache.get(tf)

        lines = [
            f"Selected timeframe: {tf}",
            f"Selected strategy: {self.strategy_box.currentText()}",
        ]

        if df is None:
            lines.append("Timeframe status: not loaded yet")
        else:
            lines.append(f"Timeframe rows: {len(df):,}")
            if len(df) > 0 and "timestamp" in df.columns:
                lines.append(f"Timeframe range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

        if self.backtest_result is not None:
            metrics = self.backtest_result.metrics
            lines.append(f"Last backtest final equity: {metrics.get('final_equity', 0):,.2f}")
            lines.append(f"Last backtest trades: {metrics.get('total_trades', 0)}")

        self.summary_box.setPlainText("\n".join(lines))

    def _build_config(self) -> BacktestConfig:
        return BacktestConfig(
            initial_capital=float(self.initial_capital.value()),
            fee_rate=float(self.fee_rate.value()),
            slippage_rate=float(self.slippage_rate.value()),
            risk_pct_per_trade=float(self.risk_pct.value()),
            allow_long=bool(self.allow_long.isChecked()),
            allow_short=bool(self.allow_short.isChecked()),
            stop_loss_pct=float(self.stop_loss_pct.value()),
            take_profit_pct=float(self.take_profit_pct.value()),
            one_position_at_a_time=True,
        )

    def _prepare_strategy_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        local = df.copy()
        local = local.sort_values("timestamp").reset_index(drop=True)

        preset = self.strategy_box.currentText()

        if preset == "EMA Cross 20/50":
            local["ema_fast"] = local["close"].ewm(span=20, adjust=False).mean()
            local["ema_slow"] = local["close"].ewm(span=50, adjust=False).mean()

            local["long_entry"] = (
                (local["ema_fast"] > local["ema_slow"]) &
                (local["ema_fast"].shift(1) <= local["ema_slow"].shift(1))
            )

            local["short_entry"] = (
                (local["ema_fast"] < local["ema_slow"]) &
                (local["ema_fast"].shift(1) >= local["ema_slow"].shift(1))
            )

        elif preset == "RSI Reversal 30/70":
            delta = local["close"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)

            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean().replace(0, pd.NA)

            rs = avg_gain / avg_loss
            local["rsi"] = 100 - (100 / (1 + rs))

            local["long_entry"] = (
                (local["rsi"] > 30) &
                (local["rsi"].shift(1) <= 30)
            )

            local["short_entry"] = (
                (local["rsi"] < 70) &
                (local["rsi"].shift(1) >= 70)
            )

        elif preset == "Breakout 20":
            local["rolling_high_20"] = local["high"].rolling(20).max().shift(1)
            local["rolling_low_20"] = local["low"].rolling(20).min().shift(1)

            local["long_entry"] = local["close"] > local["rolling_high_20"]
            local["short_entry"] = local["close"] < local["rolling_low_20"]

        else:
            raise ValueError(f"Unsupported strategy preset: {preset}")

        local["long_entry"] = local["long_entry"].fillna(False)
        local["short_entry"] = local["short_entry"].fillna(False)

        return local

    def run_backtest(self):
        tf = self.timeframe_box.currentText()
        df = self.timeframe_cache.get(tf)

        if df is None:
            QMessageBox.warning(self, "Timeframe not ready", "Selected timeframe is not loaded yet.")
            return

        if self.backtest_thread is not None:
            QMessageBox.warning(self, "Already running", "A backtest is already in progress.")
            return

        try:
            prepared_df = self._prepare_strategy_dataframe(df)
        except Exception as exc:
            QMessageBox.critical(self, "Strategy preparation failed", str(exc))
            return

        config = self._build_config()

        self.run_btn.setEnabled(False)
        self.progress.setValue(0)
        self.stage_label.setText("Stage: preparing backtest")
        self.metrics_table.setRowCount(0)
        self.trades_table.setRowCount(0)
        self.trades_table.setColumnCount(0)

        self.backtest_thread = QThread()
        self.backtest_worker = BacktestWorker(prepared_df, config)
        self.backtest_worker.moveToThread(self.backtest_thread)

        self.backtest_thread.started.connect(self.backtest_worker.run)
        self.backtest_worker.progress.connect(self.progress.setValue)
        self.backtest_worker.stage.connect(lambda t: self.stage_label.setText(f"Stage: {t}"))
        self.backtest_worker.log.connect(self.log_message.emit)
        self.backtest_worker.finished.connect(self._on_backtest_ready)
        self.backtest_worker.error.connect(self._on_backtest_error)

        self.backtest_worker.finished.connect(self.backtest_thread.quit)
        self.backtest_worker.error.connect(self.backtest_thread.quit)
        self.backtest_thread.finished.connect(self._cleanup_backtest_worker)

        self.backtest_thread.start()

    def _on_backtest_ready(self, result):
        self.backtest_result = result
        self.run_btn.setEnabled(True)
        self.progress.setValue(100)
        self.stage_label.setText("Stage: backtest complete")

        self._populate_metrics(result.metrics)
        self._populate_trades(result.trades)
        self._refresh_summary()

        self.log_message.emit(
            "INFO",
            f"Backtest complete | trades={result.metrics.get('total_trades', 0)} | final_equity={result.metrics.get('final_equity', 0):.2f}"
        )

    def _on_backtest_error(self, text: str):
        self.run_btn.setEnabled(True)
        self.progress.setValue(0)
        self.stage_label.setText("Stage: backtest failed")
        QMessageBox.critical(self, "Backtest failed", text)

    def _cleanup_backtest_worker(self):
        self.backtest_worker = None
        self.backtest_thread = None

    def _populate_metrics(self, metrics: dict):
        items = list(metrics.items())
        self.metrics_table.setRowCount(len(items))
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])

        for r, (key, value) in enumerate(items):
            self.metrics_table.setItem(r, 0, QTableWidgetItem(str(key)))
            self.metrics_table.setItem(r, 1, QTableWidgetItem(str(value)))

        self.metrics_table.resizeColumnsToContents()

    def _populate_trades(self, trades_df: pd.DataFrame):
        if trades_df is None or trades_df.empty:
            self.trades_table.setRowCount(0)
            self.trades_table.setColumnCount(0)
            return

        preview = trades_df.tail(100).reset_index(drop=True)

        self.trades_table.setRowCount(len(preview))
        self.trades_table.setColumnCount(len(preview.columns))
        self.trades_table.setHorizontalHeaderLabels(list(preview.columns))

        for r in range(len(preview)):
            for c, col in enumerate(preview.columns):
                self.trades_table.setItem(r, c, QTableWidgetItem(str(preview.iloc[r, c])))

        self.trades_table.resizeColumnsToContents()