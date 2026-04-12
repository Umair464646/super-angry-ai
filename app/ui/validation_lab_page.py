from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
)

from app.core.backtest_engine import BacktestConfig
from app.core.strategy_engine import TEMPLATES, walk_forward_validate


class ValidationLabPage(QWidget):
    log_message = pyqtSignal(str, str)
    timeframe_requested = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.timeframe_cache = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Validation Lab")
        title.setStyleSheet("font-size: 24px; font-weight: 700;")
        subtitle = QLabel(
            "Run walk-forward validation to penalize unstable or overfit strategies before they reach Results/Export."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #8a95a5;")

        top = QHBoxLayout()
        self.timeframe_box = QComboBox()
        self.timeframe_box.addItems(["1s", "5s", "15s", "30s", "1m", "5m", "15m", "1h", "4h"])
        self.timeframe_box.setCurrentText("1m")
        self.timeframe_box.currentTextChanged.connect(self._ensure_timeframe_ready)

        self.strategy_box = QComboBox()
        for t in TEMPLATES:
            self.strategy_box.addItem(t.name, t.key)

        self.run_btn = QPushButton("Run Walk-Forward")
        self.run_btn.clicked.connect(self.run_validation)

        top.addWidget(QLabel("Timeframe"))
        top.addWidget(self.timeframe_box)
        top.addWidget(QLabel("Template"))
        top.addWidget(self.strategy_box)
        top.addWidget(self.run_btn)
        top.addStretch(1)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(110)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Fold", "Rows", "Return %", "Trades", "Win %", "Max DD %"])

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addLayout(top)
        layout.addWidget(self.summary)
        layout.addWidget(self.table, 1)

        self._refresh_summary("No validation run yet.")

    def set_source_context(self, source_path: str, timeframe_cache: dict):
        _ = source_path
        self.timeframe_cache = timeframe_cache
        self._refresh_summary("Context updated.")

    def set_timeframe_dataset(self, timeframe: str, df):
        self.timeframe_cache[timeframe] = df

    def _ensure_timeframe_ready(self):
        tf = self.timeframe_box.currentText()
        if tf not in self.timeframe_cache:
            self.log_message.emit("INFO", f"Validation Lab requested timeframe build: {tf}")
            self.timeframe_requested.emit(tf)

    def _refresh_summary(self, note: str):
        tf = self.timeframe_box.currentText()
        key = self.strategy_box.currentData()
        lines = [
            f"Selected timeframe: {tf}",
            f"Selected template: {key}",
            "Validation mode: walk-forward folds (sequential, non-random).",
            "Synthetic handling: synthetic rows cannot trigger entries.",
            f"Status: {note}",
        ]
        self.summary.setPlainText("\n".join(lines))

    def run_validation(self):
        tf = self.timeframe_box.currentText()
        df = self.timeframe_cache.get(tf)
        key = self.strategy_box.currentData()

        if df is None:
            QMessageBox.warning(self, "Timeframe not ready", "Selected timeframe is not loaded yet.")
            return

        config = BacktestConfig(
            initial_capital=10_000.0,
            fee_rate=0.0004,
            slippage_rate=0.0002,
            risk_pct_per_trade=0.10,
            allow_long=True,
            allow_short=True,
            stop_loss_pct=0.01,
            take_profit_pct=0.02,
            one_position_at_a_time=True,
        )

        try:
            frame, stability = walk_forward_validate(df, key, config=config, folds=4)
        except Exception as exc:
            QMessageBox.critical(self, "Validation failed", str(exc))
            self.log_message.emit("ERROR", f"Validation failed for {key}: {exc}")
            return

        self.table.setRowCount(len(frame))
        for r in range(len(frame)):
            values = [
                int(frame.iloc[r]["fold"]),
                int(frame.iloc[r]["rows"]),
                float(frame.iloc[r]["return_pct"]),
                int(frame.iloc[r]["trades"]),
                float(frame.iloc[r]["win_rate_pct"]),
                float(frame.iloc[r]["max_drawdown_pct"]),
            ]
            for c, value in enumerate(values):
                self.table.setItem(r, c, QTableWidgetItem(str(value)))

        self.table.resizeColumnsToContents()
        self._refresh_summary(f"Validation complete. Stability score: {stability:.2f}")
        self.log_message.emit("INFO", f"Validation complete for {key} | stability={stability:.2f}")
