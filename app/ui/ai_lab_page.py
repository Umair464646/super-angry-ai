from __future__ import annotations

from PyQt6.QtCore import pyqtSignal, QThread, Qt
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
    QSplitter,
    QSpinBox,
    QCheckBox,
    QPlainTextEdit,
    QApplication,
    QFileDialog,
    QScrollArea,
    QSizePolicy,
    QFrame,
    QHeaderView,
)
import pyqtgraph as pg

from app.core.ai_worker import AIWorker
from app.core.auto_research_worker import AutoResearchWorker, ResearchRunConfig
from app.ui.ai_live_monitor import AILiveMonitorDialog
from app.ui.nn_training_window import NNTrainingWindow


FEATURE_GROUPS = [
    "EMA", "SMA", "RSI", "MACD", "ATR", "BOLLINGER",
    "VOLATILITY", "VOLUME_SPIKE", "BREAKOUT", "CANDLE_RATIOS",
    "VWAP", "MOMENTUM", "ORDER_FLOW", "ZSCORE", "DONCHIAN", "STOCHASTIC",
    "KELTNER", "ADX", "CCI", "WILLIAMS_R", "OBV", "CMF", "ICHIMOKU",
    "SUPERTREND", "FRACTAL", "MICROSTRUCTURE",
]


class AILabPage(QWidget):
    log_message = pyqtSignal(str, str)
    timeframe_requested = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.source_path = None
        self.timeframe_cache = {}

        self.ai_thread = None
        self.ai_worker = None
        self.pipeline_thread = None
        self.pipeline_worker = None

        self.ai_result = None
        self.pipeline_result = None
        self.live_monitor = None
        self.nn_window = None
        self.strategy_feed_rows: list[dict] = []
        self.last_mutation = None
        self._live_epoch_x: list[int] = []
        self._live_epoch_loss: list[float] = []
        self._live_epoch_acc: list[float] = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("AI Lab — Automated Strategy Research")
        title.setStyleSheet("font-size: 24px; font-weight: 700;")
        subtitle = QLabel(
            "Run a visible end-to-end pipeline: data profiling, auto features, strategy evolution, validation, and AI regime/confidence scoring."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #8a95a5;")

        control = QHBoxLayout()
        self.timeframe_box = QComboBox()
        self.timeframe_box.addItems(["1s", "5s", "15s", "30s", "1m", "5m", "15m", "1h", "4h"])
        self.timeframe_box.setCurrentText("1m")
        self.timeframe_box.currentTextChanged.connect(self._ensure_timeframe_ready)

        self.population_spin = QSpinBox()
        self.population_spin.setRange(6, 300)
        self.population_spin.setValue(24)

        self.generation_spin = QSpinBox()
        self.generation_spin.setRange(1, 200)
        self.generation_spin.setValue(12)

        self.auto_mode = QCheckBox("Auto mode")
        self.auto_mode.setChecked(True)
        self.model_box = QComboBox()
        self.model_box.addItems(["mlp", "logistic"])

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_pipeline)
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_pipeline)
        self.ai_only_btn = QPushButton("AI-Only Quick Run")
        self.ai_only_btn.clicked.connect(self.run_ai_only)
        self.live_btn = QPushButton("Open Live Monitor")
        self.live_btn.clicked.connect(self.open_live_monitor)

        control.addWidget(QLabel("Timeframe"))
        control.addWidget(self.timeframe_box)
        control.addWidget(QLabel("Population"))
        control.addWidget(self.population_spin)
        control.addWidget(QLabel("Generations"))
        control.addWidget(self.generation_spin)
        control.addWidget(QLabel("Model"))
        control.addWidget(self.model_box)
        control.addWidget(self.auto_mode)
        control.addWidget(self.start_btn)
        control.addWidget(self.pause_btn)
        control.addWidget(self.stop_btn)
        control.addWidget(self.ai_only_btn)
        control.addWidget(self.live_btn)
        control.addStretch(1)

        feature_row = QHBoxLayout()
        self.feature_checks = {}
        for name in FEATURE_GROUPS:
            cb = QCheckBox(name)
            cb.setChecked(True)
            self.feature_checks[name] = cb
            feature_row.addWidget(cb)
        feature_row.addStretch(1)
        feature_wrap = QWidget()
        feature_wrap.setLayout(feature_row)
        feature_scroll = QScrollArea()
        feature_scroll.setWidget(feature_wrap)
        feature_scroll.setWidgetResizable(True)
        feature_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        feature_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        feature_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        feature_scroll.setMinimumHeight(54)

        self.stage_label = QLabel("Stage: idle")
        self.stage_label.setStyleSheet("color: #8a95a5;")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(120)

        self.timeline_table = QTableWidget(0, 3)
        self.timeline_table.setHorizontalHeaderLabels(["Stage", "%", "Note"])

        self.generation_table = QTableWidget(0, 7)
        self.generation_table.setHorizontalHeaderLabels(
            ["Gen", "Best Strategy", "Fitness", "Robustness", "Stability", "Return %", "DD %"]
        )

        self.best_strategy_card = QTextEdit()
        self.best_strategy_card.setReadOnly(True)
        self.best_strategy_card.setMinimumHeight(120)

        self.tv_text = QPlainTextEdit()
        self.tv_text.setReadOnly(True)

        self.regime_table = QTableWidget(0, 2)
        self.regime_table.setHorizontalHeaderLabels(["Regime", "Rows"])
        self.conf_table = QTableWidget(0, 2)
        self.conf_table.setHorizontalHeaderLabels(["Confidence", "Rows"])
        self.pred_table = QTableWidget(0, 2)
        self.pred_table.setHorizontalHeaderLabels(["Prediction Bin", "Rows"])

        self.loss_plot = pg.PlotWidget(title="Model Loss")
        self.acc_plot = pg.PlotWidget(title="Model Accuracy")
        self.evolution_plot = pg.PlotWidget(title="Best Fitness by Generation")
        self.evolution_plot.setLabel("left", "Fitness")
        self.evolution_plot.setLabel("bottom", "Generation")

        self.timeline_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.timeline_table.verticalHeader().setVisible(False)
        self.timeline_table.setMinimumHeight(220)
        self.generation_table.verticalHeader().setVisible(False)
        self.generation_table.setMinimumHeight(220)
        self.regime_table.verticalHeader().setVisible(False)
        self.conf_table.verticalHeader().setVisible(False)
        self.pred_table.verticalHeader().setVisible(False)

        self.lifecycle_table = QTableWidget(0, 2)
        self.lifecycle_table.setHorizontalHeaderLabels(["Stage", "Count"])
        self.mutation_box = QPlainTextEdit()
        self.mutation_box.setReadOnly(True)
        self.mutation_box.setPlaceholderText("Mutation inspector waiting...")

        self.strategy_feed = QTableWidget(0, 15)
        self.strategy_feed.setHorizontalHeaderLabels([
            "ID", "Gen", "Name", "Type", "Timeframe", "Indicators", "Parameters",
            "Entry", "Exit", "Filters", "Fitness", "Robustness", "Validation", "Status", "TV"
        ])
        self.strategy_feed.itemSelectionChanged.connect(self._on_strategy_selected)
        self.strategy_feed.verticalHeader().setVisible(False)
        self.strategy_feed.setMinimumHeight(280)
        self.strategy_feed.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.strategy_text = QPlainTextEdit()
        self.strategy_text.setReadOnly(True)
        self.strategy_text.setPlaceholderText("Selected Strategy Text")
        action_row = QHBoxLayout()
        self.copy_strategy_btn = QPushButton("Copy Strategy")
        self.copy_rules_btn = QPushButton("Copy Rules")
        self.copy_report_btn = QPushButton("Copy Full Report")
        self.export_btn = QPushButton("Export txt/json")
        self.open_dna_btn = QPushButton("Open DNA Inspector")
        self.copy_strategy_btn.clicked.connect(lambda: self._copy_strategy(mode="strategy"))
        self.copy_rules_btn.clicked.connect(lambda: self._copy_strategy(mode="rules"))
        self.copy_report_btn.clicked.connect(lambda: self._copy_strategy(mode="report"))
        self.export_btn.clicked.connect(self._export_strategy_selected)
        self.open_dna_btn.clicked.connect(self._open_dna_inspector)
        for b in [self.copy_strategy_btn, self.copy_rules_btn, self.copy_report_btn, self.export_btn, self.open_dna_btn]:
            action_row.addWidget(b)
        action_row.addStretch(1)

        feed_split = QSplitter(Qt.Orientation.Horizontal)
        feed_left = QWidget()
        fl = QVBoxLayout(feed_left)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.addWidget(QLabel("Strategy Feed"))
        fl.addWidget(self.strategy_feed, 3)
        fl.addLayout(action_row)
        fl.addWidget(QLabel("Selected Strategy Text"))
        fl.addWidget(self.strategy_text, 2)
        feed_right = QWidget()
        fr = QVBoxLayout(feed_right)
        fr.setContentsMargins(0, 0, 0, 0)
        fr.addWidget(QLabel("Candidate Lifecycle"))
        fr.addWidget(self.lifecycle_table, 1)
        fr.addWidget(QLabel("Strategy DNA / Mutation Inspector"))
        fr.addWidget(self.mutation_box, 2)
        feed_split.addWidget(feed_left)
        feed_split.addWidget(feed_right)
        feed_split.setStretchFactor(0, 3)
        feed_split.setStretchFactor(1, 1)
        feed_split.setMinimumHeight(500)

        evolution_panel = self._make_panel("Pipeline Timeline", self.timeline_table)
        gen_panel = self._make_panel("Generation Evolution", self.generation_table)
        summary_panel = self._make_panel("Pipeline Summary", self.summary)

        dist_panel = QWidget()
        dist_layout = QHBoxLayout(dist_panel)
        dist_layout.setContentsMargins(0, 0, 0, 0)
        dist_layout.setSpacing(12)
        dist_layout.addWidget(self._make_panel("Regime Distribution", self.regime_table), 1)
        dist_layout.addWidget(self._make_panel("Confidence Distribution", self.conf_table), 1)
        dist_layout.addWidget(self._make_panel("Prediction Distribution", self.pred_table), 1)

        charts_split = QSplitter(Qt.Orientation.Vertical)
        charts_split.addWidget(self._make_panel("Best Fitness by Generation", self.evolution_plot))
        charts_split.addWidget(self._make_panel("Model Loss", self.loss_plot))
        charts_split.addWidget(self._make_panel("Model Accuracy", self.acc_plot))
        charts_split.setStretchFactor(0, 2)
        charts_split.setStretchFactor(1, 1)
        charts_split.setStretchFactor(2, 1)
        charts_split.setMinimumHeight(540)

        strategy_details = QWidget()
        sd = QHBoxLayout(strategy_details)
        sd.setContentsMargins(0, 0, 0, 0)
        sd.setSpacing(12)
        sd.addWidget(self._make_panel("Current Best Strategy", self.best_strategy_card), 1)
        sd.addWidget(self._make_panel("TradingView Replication Package", self.tv_text), 1)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addLayout(control)
        layout.addWidget(feature_scroll)
        layout.addWidget(self.stage_label)
        layout.addWidget(self.progress)
        layout.addWidget(summary_panel)

        content_host = QWidget()
        content_layout = QVBoxLayout(content_host)
        content_layout.setContentsMargins(6, 6, 6, 6)
        content_layout.setSpacing(14)
        content_layout.addWidget(evolution_panel)
        content_layout.addWidget(gen_panel)
        content_layout.addWidget(dist_panel)
        content_layout.addWidget(charts_split)
        content_layout.addWidget(strategy_details)
        content_layout.addWidget(feed_split)
        content_layout.addStretch(1)

        content_scroll = QScrollArea()
        content_scroll.setWidgetResizable(True)
        content_scroll.setWidget(content_host)
        content_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        content_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        layout.addWidget(content_scroll, 1)

        self._set_buttons_running(False)
        self._refresh_summary()

    def _make_panel(self, title: str, child: QWidget) -> QWidget:
        frame = QFrame()
        frame.setObjectName("AIPanelCard")
        frame.setStyleSheet(
            """
            QFrame#AIPanelCard {
                background: #0b111a;
                border: 1px solid #1d2a3b;
                border-radius: 10px;
            }
            """
        )
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)
        head = QLabel(title)
        head.setStyleSheet("font-size:14px; font-weight:700; color:#d6eaff;")
        lay.addWidget(head)
        lay.addWidget(child)
        return frame

    def _selected_features(self):
        return [name for name, cb in self.feature_checks.items() if cb.isChecked()]

    def set_source_context(self, source_path: str, timeframe_cache: dict):
        self.source_path = source_path
        self.timeframe_cache = timeframe_cache
        self._refresh_summary()

    def set_timeframe_dataset(self, timeframe: str, df):
        self.timeframe_cache[timeframe] = df
        self._refresh_summary()

    def set_dataframe(self, df):
        if df is not None:
            self.timeframe_cache["1s"] = df
            self._refresh_summary()

    def _ensure_timeframe_ready(self):
        tf = self.timeframe_box.currentText()
        if tf not in self.timeframe_cache:
            self.log_message.emit("INFO", f"AI Lab requested timeframe build: {tf}")
            self.timeframe_requested.emit(tf)
        self._refresh_summary()

    def _refresh_summary(self):
        tf = self.timeframe_box.currentText()
        df = self.timeframe_cache.get(tf)
        lines = [
            f"Selected timeframe: {tf}",
            f"Selected feature groups: {', '.join(self._selected_features()) or 'none'}",
            f"Population size: {self.population_spin.value()} | Generations: {self.generation_spin.value()}",
            "Pipeline: data profile → feature engineering → strategy evolution → validation → AI scoring → TradingView output.",
        ]
        if df is None:
            lines.append("Timeframe status: not loaded yet")
        else:
            lines.append(f"Rows available: {len(df):,} | Columns: {len(df.columns):,}")
        if self.pipeline_result is not None:
            top = self.pipeline_result["top_strategy"]
            lines.append(f"Last best strategy: {top['best_strategy']} (gen {int(top['generation'])})")
            lines.append(f"Last best fitness: {float(top['fitness']):.2f}")

        self.summary.setPlainText("\n".join(lines))

    def _set_buttons_running(self, running: bool):
        self.start_btn.setEnabled(not running)
        self.ai_only_btn.setEnabled(not running)
        self.live_btn.setEnabled(True)
        self.pause_btn.setEnabled(running)
        self.stop_btn.setEnabled(running)

    def open_live_monitor(self):
        if self.live_monitor is None:
            self.live_monitor = AILiveMonitorDialog(self)
        self.live_monitor.show()
        self.live_monitor.raise_()
        self.live_monitor.activateWindow()

    def open_nn_window(self):
        if self.nn_window is None:
            self.nn_window = NNTrainingWindow(self)
        self.nn_window.show()
        self.nn_window.raise_()
        self.nn_window.activateWindow()

    def _preview_architecture(self, model_type: str) -> str:
        if model_type == "logistic":
            return "Input(7) -> Logistic(1)"
        return "Input(7) -> Dense(14, tanh) -> Dense(1, sigmoid)"

    def start_pipeline(self):
        tf = self.timeframe_box.currentText()
        df = self.timeframe_cache.get(tf)
        if df is None:
            QMessageBox.warning(self, "Timeframe not ready", "Selected timeframe is not loaded yet.")
            return
        if self.pipeline_thread is not None:
            QMessageBox.warning(self, "Already running", "Pipeline is already running.")
            return

        selected = self._selected_features()
        if not selected:
            QMessageBox.warning(self, "No features selected", "Select at least one feature group.")
            return

        self._reset_run_views()
        self._set_buttons_running(True)
        self.stage_label.setText("Stage: starting automated pipeline")
        self.open_live_monitor()
        self.open_nn_window()
        if self.nn_window is not None:
            self.nn_window.reset_run()
            self.nn_window.set_architecture(self._preview_architecture(self.model_box.currentText()))

        config = ResearchRunConfig(
            selected_features=selected,
            generations=int(self.generation_spin.value()),
            population_top_k=int(self.population_spin.value()),
            max_variants_per_generation=max(250, int(self.population_spin.value()) * 25),
            validation_folds=4,
            model_type=self.model_box.currentText(),
        )

        self.pipeline_thread = QThread()
        self.pipeline_worker = AutoResearchWorker(df, config)
        self.pipeline_worker.moveToThread(self.pipeline_thread)

        self.pipeline_thread.started.connect(self.pipeline_worker.run)
        self.pipeline_worker.progress.connect(self.progress.setValue)
        self.pipeline_worker.stage.connect(lambda t: self.stage_label.setText(f"Stage: {t}"))
        self.pipeline_worker.log.connect(self.log_message.emit)
        self.pipeline_worker.timeline.connect(self._on_timeline)
        self.pipeline_worker.generation.connect(self._on_generation)
        self.pipeline_worker.candidate_test.connect(self._on_candidate_progress)
        self.pipeline_worker.ai_epoch.connect(self._on_ai_epoch)
        self.pipeline_worker.strategy_event.connect(self._on_strategy_event)
        self.pipeline_worker.lifecycle_event.connect(self._on_lifecycle_event)
        self.pipeline_worker.mutation_event.connect(self._on_mutation_event)
        self.pipeline_worker.finished.connect(self._on_pipeline_finished)
        self.pipeline_worker.error.connect(self._on_pipeline_error)

        if self.live_monitor is not None:
            self.pipeline_worker.progress.connect(self.live_monitor.on_progress)
            self.pipeline_worker.stage.connect(self.live_monitor.on_stage)
            self.pipeline_worker.log.connect(self.live_monitor.on_log)
            self.pipeline_worker.timeline.connect(self.live_monitor.on_timeline)
            self.pipeline_worker.generation.connect(self.live_monitor.on_generation)
            self.pipeline_worker.candidate_test.connect(self.live_monitor.on_candidate)

        self.pipeline_worker.finished.connect(self.pipeline_thread.quit)
        self.pipeline_worker.error.connect(self.pipeline_thread.quit)
        self.pipeline_thread.finished.connect(self._cleanup_pipeline)

        self.pipeline_thread.start()

    def toggle_pause(self):
        if self.pipeline_worker is None:
            return
        paused = self.pause_btn.text() == "Pause"
        self.pipeline_worker.set_paused(paused)
        self.pause_btn.setText("Resume" if paused else "Pause")
        self.log_message.emit("INFO", "Pipeline paused" if paused else "Pipeline resumed")

    def stop_pipeline(self):
        if self.pipeline_worker is not None:
            self.pipeline_worker.cancel()
            self.log_message.emit("WARN", "Stop requested for automated pipeline")

    def run_ai_only(self):
        tf = self.timeframe_box.currentText()
        df = self.timeframe_cache.get(tf)
        if df is None:
            QMessageBox.warning(self, "Timeframe not ready", "Selected timeframe is not loaded yet.")
            return
        if self.ai_thread is not None:
            QMessageBox.warning(self, "Already running", "AI-only run already in progress.")
            return

        self._set_buttons_running(True)
        self.ai_thread = QThread()
        self.open_nn_window()
        if self.nn_window is not None:
            self.nn_window.reset_run()
            self.nn_window.set_architecture(self._preview_architecture(self.model_box.currentText()))
        self.ai_worker = AIWorker(df, model_type=self.model_box.currentText())
        self.ai_worker.moveToThread(self.ai_thread)

        self.ai_thread.started.connect(self.ai_worker.run)
        self.ai_worker.progress.connect(self.progress.setValue)
        self.ai_worker.stage.connect(lambda t: self.stage_label.setText(f"Stage: {t}"))
        self.ai_worker.log.connect(self.log_message.emit)
        self.ai_worker.finished.connect(self._on_ai_only_ready)
        self.ai_worker.error.connect(self._on_pipeline_error)
        self.ai_worker.epoch.connect(self._on_ai_epoch)

        self.ai_worker.finished.connect(self.ai_thread.quit)
        self.ai_worker.error.connect(self.ai_thread.quit)
        self.ai_thread.finished.connect(self._cleanup_ai_only)

        self.ai_thread.start()

    def _on_timeline(self, stage_name: str, pct: int, note: str):
        row = self.timeline_table.rowCount()
        self.timeline_table.insertRow(row)
        self.timeline_table.setItem(row, 0, QTableWidgetItem(stage_name))
        self.timeline_table.setItem(row, 1, QTableWidgetItem(str(pct)))
        self.timeline_table.setItem(row, 2, QTableWidgetItem(note))
        self.timeline_table.scrollToBottom()

    def _on_generation(self, gen: int, survivors: int, best_fitness: float, population: int):
        row = self.generation_table.rowCount()
        self.generation_table.insertRow(row)
        self.generation_table.setItem(row, 0, QTableWidgetItem(str(gen)))
        self.generation_table.setItem(row, 1, QTableWidgetItem("best candidate"))
        self.generation_table.setItem(row, 2, QTableWidgetItem(f"{best_fitness:.2f}"))
        self.generation_table.setItem(row, 3, QTableWidgetItem("-"))
        self.generation_table.setItem(row, 4, QTableWidgetItem("-"))
        self.generation_table.setItem(row, 5, QTableWidgetItem(f"survivors={survivors}"))
        self.generation_table.setItem(row, 6, QTableWidgetItem(f"population={population}"))
        if self.nn_window is not None:
            self.nn_window.on_generation(gen, survivors, population)

    def _on_candidate_progress(self, gen: int, done: int, total: int, family: str):
        self.stage_label.setText(
            f"Stage: testing candidates | gen {gen} | {family} {done}/{total}"
        )
        if self.nn_window is not None:
            self.nn_window.on_candidate(gen, done, total, family)

    def _on_pipeline_finished(self, payload):
        self.pipeline_result = payload
        self._set_buttons_running(False)
        self.pause_btn.setText("Pause")
        self.stage_label.setText("Stage: automated research complete")

        best_df = payload["best_by_generation"]
        self.generation_table.setRowCount(0)
        for r in range(len(best_df)):
            row = best_df.iloc[r]
            vals = [
                int(row["generation"]),
                str(row["best_strategy"]),
                f"{row['fitness']:.2f}",
                f"{row['robustness_score']:.2f}",
                f"{row['stability_score']:.2f}",
                f"{row['test_return_pct']:.2f}",
                f"{row['test_max_drawdown_pct']:.2f}",
            ]
            for c, v in enumerate(vals):
                self.generation_table.setItem(r, c, QTableWidgetItem(str(v)))

        x = best_df["generation"].astype(float).tolist()
        y = best_df["fitness"].astype(float).tolist()
        self.evolution_plot.clear()
        self.evolution_plot.plot(x, y, pen=pg.mkPen(color="#7cfc00", width=2), symbol="o")

        ai = payload["ai"]
        self._populate_two_col_table(self.regime_table, ai.regime_counts)
        self._populate_two_col_table(self.conf_table, ai.confidence_distribution)
        self._populate_two_col_table(self.pred_table, ai.prediction_distribution)
        self.loss_plot.clear()
        self.acc_plot.clear()
        if ai.loss_curve:
            self.loss_plot.plot(list(range(1, len(ai.loss_curve) + 1)), ai.loss_curve, pen=pg.mkPen("#ff6b6b", width=2))
        if ai.accuracy_curve:
            self.acc_plot.plot(list(range(1, len(ai.accuracy_curve) + 1)), ai.accuracy_curve, pen=pg.mkPen("#00d4ff", width=2))

        top = payload["top_strategy"]
        self.best_strategy_card.setPlainText(
            "\n".join([
                f"Best strategy: {top['best_strategy']}",
                f"Template: {top['template_key']}",
                f"Generation: {int(top['generation'])}",
                f"Fitness: {float(top['fitness']):.2f}",
                f"Robustness: {float(top['robustness_score']):.2f}",
                f"Stability: {float(top['stability_score']):.2f}",
                f"Return %: {float(top['test_return_pct']):.2f}",
                f"Win rate %: {float(top['test_win_rate_pct']):.2f}",
                f"Max DD %: {float(top['test_max_drawdown_pct']):.2f}",
                f"Params: {top['params']}",
            ])
        )
        self.tv_text.setPlainText(payload["tradingview_text"])
        if self.nn_window is not None:
            self.nn_window.set_architecture(payload["ai"].nn_architecture)
            self.nn_window.on_finished()

        self._refresh_summary()
        self.log_message.emit("INFO", "Automated research pipeline complete")
        if self.live_monitor is not None:
            self.live_monitor.on_finished()

    def _on_ai_only_ready(self, result):
        self.ai_result = result
        self._set_buttons_running(False)
        self.stage_label.setText("Stage: AI-only run complete")
        self._populate_two_col_table(self.regime_table, result.regime_counts)
        self._populate_two_col_table(self.conf_table, result.confidence_distribution)
        self._populate_two_col_table(self.pred_table, result.prediction_distribution)
        self.loss_plot.clear()
        self.acc_plot.clear()
        self.loss_plot.plot(list(range(1, len(result.loss_curve) + 1)), result.loss_curve, pen=pg.mkPen("#ff6b6b", width=2))
        self.acc_plot.plot(list(range(1, len(result.accuracy_curve) + 1)), result.accuracy_curve, pen=pg.mkPen("#00d4ff", width=2))
        if self.nn_window is not None:
            self.nn_window.set_architecture(result.nn_architecture)
            self.nn_window.on_finished()

    def _on_ai_epoch(self, epoch: int, total: int, loss: float, acc: float, extra: dict | None = None):
        if self.nn_window is not None:
            self.nn_window.on_epoch(epoch, total, loss, acc, extra=extra or {})
        self._live_epoch_x.append(epoch)
        self._live_epoch_loss.append(loss)
        self._live_epoch_acc.append(acc)
        self.loss_plot.clear()
        self.acc_plot.clear()
        self.loss_plot.plot(self._live_epoch_x, self._live_epoch_loss, pen=pg.mkPen("#ff6b6b", width=2))
        self.acc_plot.plot(self._live_epoch_x, self._live_epoch_acc, pen=pg.mkPen("#00d4ff", width=2))

    def _on_pipeline_error(self, text: str):
        self._set_buttons_running(False)
        self.pause_btn.setText("Pause")
        self.stage_label.setText("Stage: failed")
        QMessageBox.critical(self, "Pipeline failed", text)

    def _cleanup_pipeline(self):
        self.pipeline_worker = None
        self.pipeline_thread = None

    def _cleanup_ai_only(self):
        self.ai_worker = None
        self.ai_thread = None

    def _populate_two_col_table(self, table: QTableWidget, data: dict):
        table.setRowCount(0)
        for k, v in data.items():
            r = table.rowCount()
            table.insertRow(r)
            table.setItem(r, 0, QTableWidgetItem(str(k)))
            table.setItem(r, 1, QTableWidgetItem(str(v)))
        table.resizeColumnsToContents()

    def _reset_run_views(self):
        self.progress.setValue(0)
        self.timeline_table.setRowCount(0)
        self.generation_table.setRowCount(0)
        self.best_strategy_card.clear()
        self.tv_text.clear()
        self.loss_plot.clear()
        self.acc_plot.clear()
        self.evolution_plot.clear()
        self.regime_table.setRowCount(0)
        self.conf_table.setRowCount(0)
        self.pred_table.setRowCount(0)
        self.strategy_feed_rows.clear()
        self._live_epoch_x.clear()
        self._live_epoch_loss.clear()
        self._live_epoch_acc.clear()
        self.strategy_feed.setRowCount(0)
        self.strategy_text.clear()
        self.mutation_box.clear()
        self.lifecycle_table.setRowCount(0)
        if self.nn_window is not None:
            self.nn_window.reset_run()

    def _on_strategy_event(self, ev: dict):
        self.strategy_feed_rows.append(ev)
        r = self.strategy_feed.rowCount()
        self.strategy_feed.insertRow(r)
        vals = [
            ev["strategy_id"], ev["generation"], ev["name"], ev["type"], ev["timeframe"],
            ev["indicators"], str(ev["parameters"]), ev["entry_logic"], ev["exit_logic"], ev["filters"],
            f"{ev['fitness']:.2f}", f"{ev['robustness']:.2f}", f"{ev['validation_score']:.2f}", ev["status"], ev["tradingview_ready"]
        ]
        for c, v in enumerate(vals):
            self.strategy_feed.setItem(r, c, QTableWidgetItem(str(v)))
        self.strategy_feed.scrollToBottom()

    def _on_lifecycle_event(self, counts: dict):
        self.lifecycle_table.setRowCount(0)
        for k, v in counts.items():
            r = self.lifecycle_table.rowCount()
            self.lifecycle_table.insertRow(r)
            self.lifecycle_table.setItem(r, 0, QTableWidgetItem(str(k)))
            self.lifecycle_table.setItem(r, 1, QTableWidgetItem(str(v)))

    def _on_mutation_event(self, mutation: dict):
        self.last_mutation = mutation
        self.mutation_box.setPlainText(
            "\n".join([
                f"Parent: {mutation.get('parent_id', 'n/a')}",
                f"Child: {mutation.get('child_id', 'n/a')}",
                f"Mutation Type: {mutation.get('mutation_type', 'n/a')}",
                "Changes:",
                *[f"- {x}" for x in mutation.get("changes", [])],
                f"Fitness Δ: {mutation.get('fitness_delta', 0.0):.2f}",
                f"Robustness Δ: {mutation.get('robustness_delta', 0.0):.2f}",
            ])
        )

    def _selected_strategy_row(self) -> dict | None:
        row = self.strategy_feed.currentRow()
        if row < 0 or row >= len(self.strategy_feed_rows):
            return None
        return self.strategy_feed_rows[row]

    def _on_strategy_selected(self):
        ev = self._selected_strategy_row()
        if ev is None:
            return
        m = ev.get("metrics", {})
        indicators_block = str(ev["indicators"]).replace(", ", "\n- ")
        text = (
            f"Strategy ID: {ev['strategy_id']}\n"
            f"Name: {ev['name']}\n"
            f"Type: {ev['type']}\n"
            f"Timeframe: {ev['timeframe']}\n\n"
            f"Indicators:\n- {indicators_block}\n\n"
            f"Entry Logic:\n- {ev['entry_logic']}\n\n"
            f"Exit Logic:\n- {ev['exit_logic']}\n\n"
            f"Filters:\n- {ev['filters']}\n\n"
            f"Fitness: {ev['fitness']:.2f}\n"
            f"Robustness: {ev['robustness']:.2f}\n"
            f"Validation Score: {ev['validation_score']:.2f}\n"
            f"Max Drawdown: {m.get('drawdown_pct', 0.0):.2f}%\n"
            f"Return: {m.get('return_pct', 0.0):.2f}%\n"
            f"Trade Count: {m.get('trades', 0)}\n"
            f"TradingView Replicable: {ev['tradingview_ready']}\n"
        )
        self.strategy_text.setPlainText(text)

    def _copy_strategy(self, mode: str = "strategy"):
        ev = self._selected_strategy_row()
        if ev is None:
            return
        if mode == "rules":
            out = f"Entry: {ev['entry_logic']}\nExit: {ev['exit_logic']}\nFilters: {ev['filters']}"
        elif mode == "report":
            out = self.strategy_text.toPlainText() + f"\nParameters:\n{ev['parameters']}"
        else:
            out = self.strategy_text.toPlainText()
        QApplication.clipboard().setText(out)

    def _export_strategy_selected(self):
        ev = self._selected_strategy_row()
        if ev is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Strategy", "strategy_report.txt", "Text (*.txt);;JSON (*.json)")
        if not path:
            return
        if path.endswith(".json"):
            import json
            with open(path, "w", encoding="utf-8") as f:
                json.dump(ev, f, indent=2)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.strategy_text.toPlainText() + "\n\nParameters:\n" + str(ev["parameters"]))

    def _open_dna_inspector(self):
        if self.last_mutation:
            QMessageBox.information(self, "DNA Inspector", self.mutation_box.toPlainText() or "No mutation data yet.")
