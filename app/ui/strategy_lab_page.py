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
    QPlainTextEdit,
)
import pyqtgraph as pg

from app.core.backtest_engine import BacktestConfig
from app.core.strategy_engine import (
    TEMPLATES,
    evaluate_template,
    evolve_templates,
    tradingview_strategy_text,
)


class StrategyLabPage(QWidget):
    log_message = pyqtSignal(str, str)
    timeframe_requested = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.timeframe_cache = {}
        self.latest_results = []
        self.evolution_all = None
        self.evolution_top = None
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
        self.evolve_btn = QPushButton("Run Evolution Sweep")
        self.evolve_btn.clicked.connect(self.run_evolution)

        top.addWidget(QLabel("Timeframe"))
        top.addWidget(self.timeframe_box)
        top.addWidget(self.generate_btn)
        top.addWidget(self.evolve_btn)
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
        self.tv_export = QPlainTextEdit()
        self.tv_export.setReadOnly(True)
        self.evolution_plot = pg.PlotWidget(title="Evolution Fitness (Top Variants)")
        self.evolution_plot.setLabel("left", "Fitness")
        self.evolution_plot.setLabel("bottom", "Rank")

        splitter = QSplitter()
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QLabel("Selected Strategy Details"))
        right_layout.addWidget(self.details, 2)
        right_layout.addWidget(QLabel("TradingView Replication Text"))
        right_layout.addWidget(self.tv_export, 2)
        right_layout.addWidget(self.evolution_plot, 2)

        splitter.addWidget(self.table)
        splitter.addWidget(right)
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
        if self.evolution_top is not None and not self.evolution_top.empty:
            lines.append(f"Evolution top variants available: {len(self.evolution_top)}")

        self.summary.setPlainText("\n".join(lines))

    def run_generation(self):
        tf = self.timeframe_box.currentText()
        df = self.timeframe_cache.get(tf)

        if df is None:
            QMessageBox.warning(self, "Timeframe not ready", "Selected timeframe is not loaded yet.")
            return

        self.generate_btn.setEnabled(False)
        self.evolve_btn.setEnabled(False)
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
                result = evaluate_template(df, template.key, config=config)
                self.latest_results.append(result)
            except Exception as exc:
                errors.append(f"{template.name}: {exc}")

        self.latest_results.sort(key=lambda r: r["robustness_score"], reverse=True)
        self._populate_results_table()
        self._refresh_summary()
        self.generate_btn.setEnabled(True)
        self.evolve_btn.setEnabled(True)

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

    def run_evolution(self):
        tf = self.timeframe_box.currentText()
        df = self.timeframe_cache.get(tf)
        if df is None:
            QMessageBox.warning(self, "Timeframe not ready", "Selected timeframe is not loaded yet.")
            return

        self.generate_btn.setEnabled(False)
        self.evolve_btn.setEnabled(False)
        self.evolution_plot.clear()

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
            all_variants, top_variants = evolve_templates(df, config=config, top_k=12)
        except Exception as exc:
            self.generate_btn.setEnabled(True)
            self.evolve_btn.setEnabled(True)
            QMessageBox.critical(self, "Evolution failed", str(exc))
            return

        self.evolution_all = all_variants
        self.evolution_top = top_variants
        self.latest_results = []
        self.table.setRowCount(len(top_variants))

        for r in range(len(top_variants)):
            row = top_variants.iloc[r]
            values = [
                str(row["strategy"]),
                f"{row['robustness_score']:.2f}",
                "",
                f"{row['test_return_pct']:.2f}",
                f"{row['test_max_drawdown_pct']:.2f}",
                f"{row['test_win_rate_pct']:.2f}",
                str(int(row["test_trades"])),
                str(row["template_key"]),
            ]
            for c, value in enumerate(values):
                self.table.setItem(r, c, QTableWidgetItem(value))

        self.table.resizeColumnsToContents()
        self._plot_evolution(top_variants)
        self._refresh_summary()
        self.generate_btn.setEnabled(True)
        self.evolve_btn.setEnabled(True)
        self.log_message.emit("INFO", f"Evolution sweep complete: {len(all_variants)} variants scored")

        if len(top_variants) > 0:
            self.table.selectRow(0)

    def _plot_evolution(self, top_variants):
        if top_variants is None or top_variants.empty:
            return
        y = top_variants["fitness"].astype(float).tolist()
        x = list(range(1, len(y) + 1))
        self.evolution_plot.plot(x, y, pen=pg.mkPen(color="#7cfc00", width=2), symbol="o")

    def _render_selected_details(self):
        row = self.table.currentRow()
        if row < 0:
            return

        if self.latest_results and row < len(self.latest_results):
            item = self.latest_results[row]
            template = item["template"]
            test = item["test"].metrics
            chosen_params = item.get("params", template.params)
            robustness = item["robustness_score"]
        elif self.evolution_top is not None and row < len(self.evolution_top):
            evo = self.evolution_top.iloc[row]
            template = next(t for t in TEMPLATES if t.key == evo["template_key"])
            chosen_params = dict(evo["params"])
            test = {
                "total_return_pct": float(evo["test_return_pct"]),
                "total_trades": int(evo["test_trades"]),
                "win_rate_pct": float(evo["test_win_rate_pct"]),
                "max_drawdown_pct": float(evo["test_max_drawdown_pct"]),
            }
            robustness = float(evo["robustness_score"])
        else:
            return

        details = [
            f"Strategy: {template.name}",
            f"Template key: {template.key}",
            f"Robustness score: {robustness:.2f}",
            "",
            "Indicators:",
            *[f"- {x}" for x in template.indicators],
            "",
            "Parameters:",
            *[f"- {k}: {v}" for k, v in chosen_params.items()],
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
        self.tv_export.setPlainText(tradingview_strategy_text(template.key, chosen_params))
