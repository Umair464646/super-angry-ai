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
    QSplitter,
)

from app.core.backtest_engine import BacktestConfig
from app.core.strategy_engine import TEMPLATES, evaluate_template


class StrategyLabPage(QWidget):
    log_message = pyqtSignal(str, str)
    timeframe_requested = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.timeframe_cache = {}
        self.latest_results = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Strategy Lab")
        title.setStyleSheet("font-size: 24px; font-weight: 700;")
        subtitle = QLabel(
            "Generate transparent rule-based strategy candidates, score them with train/test backtests, and keep only TradingView-replicable logic."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #8a95a5;")

        top = QHBoxLayout()
        self.timeframe_box = QComboBox()
        self.timeframe_box.addItems(["1s", "5s", "15s", "30s", "1m", "5m", "15m", "1h", "4h"])
        self.timeframe_box.setCurrentText("1m")
        self.timeframe_box.currentTextChanged.connect(self._ensure_timeframe_ready)

        self.generate_btn = QPushButton("Generate + Score Candidates")
        self.generate_btn.clicked.connect(self.run_generation)

        top.addWidget(QLabel("Timeframe"))
        top.addWidget(self.timeframe_box)
        top.addWidget(self.generate_btn)
        top.addStretch(1)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(120)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            [
                "Strategy",
                "Robustness",
                "Full Return %",
                "Test Return %",
                "Test DD %",
                "Test Win %",
                "Test Trades",
                "Template Key",
            ]
        )
        self.table.itemSelectionChanged.connect(self._render_selected_details)

        self.details = QTextEdit()
        self.details.setReadOnly(True)

        splitter = QSplitter()
        splitter.addWidget(self.table)
        splitter.addWidget(self.details)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addLayout(top)
        layout.addWidget(self.summary)
        layout.addWidget(splitter, 1)

        self._refresh_summary()

    def set_source_context(self, source_path: str, timeframe_cache: dict):
        _ = source_path
        self.timeframe_cache = timeframe_cache
        self._refresh_summary()

    def set_timeframe_dataset(self, timeframe: str, df):
        self.timeframe_cache[timeframe] = df
        self._refresh_summary()

    def _ensure_timeframe_ready(self):
        tf = self.timeframe_box.currentText()
        if tf not in self.timeframe_cache:
            self.log_message.emit("INFO", f"Strategy Lab requested timeframe build: {tf}")
            self.timeframe_requested.emit(tf)
        self._refresh_summary()

    def _refresh_summary(self):
        tf = self.timeframe_box.currentText()
        df = self.timeframe_cache.get(tf)
        lines = [
            f"Selected timeframe: {tf}",
            f"Candidate templates available: {len(TEMPLATES)}",
            "Synthetic policy: signal triggers disabled when synthetic == 1.",
        ]

        if df is None:
            lines.append("Timeframe status: not loaded yet")
        else:
            lines.append(f"Rows available: {len(df):,}")
            lines.append(f"Columns available: {len(df.columns):,}")

        if self.latest_results:
            lines.append(f"Last generation run: {len(self.latest_results)} candidates scored")

        self.summary.setPlainText("\n".join(lines))

    def run_generation(self):
        tf = self.timeframe_box.currentText()
        df = self.timeframe_cache.get(tf)

        if df is None:
            QMessageBox.warning(self, "Timeframe not ready", "Selected timeframe is not loaded yet.")
            return

        self.generate_btn.setEnabled(False)
        self.table.setRowCount(0)
        self.latest_results = []
        errors = []

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

        for template in TEMPLATES:
            try:
                result = evaluate_template(df, template.key, config)
                self.latest_results.append(result)
            except Exception as exc:
                errors.append(f"{template.name}: {exc}")

        self.latest_results.sort(key=lambda r: r["robustness_score"], reverse=True)
        self._populate_results_table()
        self._refresh_summary()
        self.generate_btn.setEnabled(True)

        if errors:
            self.log_message.emit("WARN", f"Strategy generation completed with {len(errors)} template errors")
            self.details.setPlainText("\n".join(["Generation warnings:", *errors]))
        else:
            self.log_message.emit("INFO", f"Strategy generation complete: {len(self.latest_results)} candidates scored")

    def _populate_results_table(self):
        self.table.setRowCount(len(self.latest_results))

        for r, item in enumerate(self.latest_results):
            template = item["template"]
            full = item["full"].metrics
            test = item["test"].metrics

            values = [
                template.name,
                f"{item['robustness_score']:.2f}",
                f"{full['total_return_pct']:.2f}",
                f"{test['total_return_pct']:.2f}",
                f"{test['max_drawdown_pct']:.2f}",
                f"{test['win_rate_pct']:.2f}",
                str(test['total_trades']),
                template.key,
            ]

            for c, value in enumerate(values):
                self.table.setItem(r, c, QTableWidgetItem(value))

        self.table.resizeColumnsToContents()

        if self.latest_results:
            self.table.selectRow(0)

    def _render_selected_details(self):
        row = self.table.currentRow()
        if row < 0 or row >= len(self.latest_results):
            return

        item = self.latest_results[row]
        template = item["template"]
        test = item["test"].metrics

        details = [
            f"Strategy: {template.name}",
            f"Template key: {template.key}",
            f"Robustness score: {item['robustness_score']:.2f}",
            "",
            "Indicators:",
            *[f"- {x}" for x in template.indicators],
            "",
            "Parameters:",
            *[f"- {k}: {v}" for k, v in template.params.items()],
            "",
            f"Entry logic: {template.entry_logic}",
            f"Exit logic: {template.exit_logic}",
            f"Filters: {template.filters}",
            "",
            "Test slice metrics:",
            f"- return_pct: {test['total_return_pct']:.2f}",
            f"- trades: {test['total_trades']}",
            f"- win_rate_pct: {test['win_rate_pct']:.2f}",
            f"- max_drawdown_pct: {test['max_drawdown_pct']:.2f}",
            "",
            "TradingView replication note:",
            "- All logic above is deterministic and indicator/threshold based.",
            "- No black-box hidden rules are used in this phase.",
        ]

        self.details.setPlainText("\n".join(details))
