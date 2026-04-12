from __future__ import annotations

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
    QSplitter,
    QSpinBox,
    QCheckBox,
    QPlainTextEdit,
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

        top_split = QSplitter()
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("Pipeline Timeline"))
        ll.addWidget(self.timeline_table)
        ll.addWidget(QLabel("Generation Evolution"))
        ll.addWidget(self.generation_table)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QLabel("Current Best Strategy"))
        rl.addWidget(self.best_strategy_card)
        rl.addWidget(QLabel("TradingView Replication Package"))
        rl.addWidget(self.tv_text)

        top_split.addWidget(left)
        top_split.addWidget(right)
        top_split.setStretchFactor(0, 2)
        top_split.setStretchFactor(1, 2)

        lower_split = QSplitter()
        left_lower = QWidget()
        lll = QVBoxLayout(left_lower)
        lll.setContentsMargins(0, 0, 0, 0)
        lll.addWidget(QLabel("Regime Distribution"))
        lll.addWidget(self.regime_table)
        lll.addWidget(QLabel("Confidence Distribution"))
        lll.addWidget(self.conf_table)
        lll.addWidget(QLabel("Prediction Distribution"))
        lll.addWidget(self.pred_table)

        right_lower = QWidget()
        rll = QVBoxLayout(right_lower)
        rll.setContentsMargins(0, 0, 0, 0)
        rll.addWidget(self.evolution_plot)
        rll.addWidget(self.loss_plot)
        rll.addWidget(self.acc_plot)

        lower_split.addWidget(left_lower)
        lower_split.addWidget(right_lower)
        lower_split.setStretchFactor(0, 1)
        lower_split.setStretchFactor(1, 2)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addLayout(control)
        layout.addLayout(feature_row)
        layout.addWidget(self.stage_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.summary)
        layout.addWidget(top_split, 2)
        layout.addWidget(lower_split, 2)

        self._set_buttons_running(False)
        self._refresh_summary()

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
        if self.nn_window is not None:
            self.nn_window.reset_run()
