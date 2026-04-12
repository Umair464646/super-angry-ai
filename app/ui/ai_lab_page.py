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
)
import pyqtgraph as pg

from app.core.ai_worker import AIWorker


class AILabPage(QWidget):
    log_message = pyqtSignal(str, str)
    timeframe_requested = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.source_path = None
        self.timeframe_cache = {}
        self.ai_thread = None
        self.ai_worker = None
        self.ai_result = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("AI Lab")
        title.setStyleSheet("font-size: 24px; font-weight: 700;")

        subtitle = QLabel(
            "Visible AI layer: regime classification, setup confidence scoring, and neural-style training curves (loss/accuracy)."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #8a95a5;")

        top = QHBoxLayout()
        self.timeframe_box = QComboBox()
        self.timeframe_box.addItems(["1s", "5s", "15s", "30s", "1m", "5m", "15m", "1h", "4h"])
        self.timeframe_box.setCurrentText("1m")
        self.timeframe_box.currentTextChanged.connect(self._ensure_timeframe_ready)

        self.run_btn = QPushButton("Run AI Analysis")
        self.run_btn.clicked.connect(self.run_ai)

        top.addWidget(QLabel("Timeframe"))
        top.addWidget(self.timeframe_box)
        top.addWidget(self.run_btn)
        top.addStretch(1)

        self.stage_label = QLabel("Stage: idle")
        self.stage_label.setStyleSheet("color: #8a95a5;")

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(130)

        self.regime_table = QTableWidget(0, 2)
        self.regime_table.setHorizontalHeaderLabels(["Regime", "Rows"])

        self.conf_table = QTableWidget(0, 2)
        self.conf_table.setHorizontalHeaderLabels(["Confidence Band", "Rows"])

        self.setups_table = QTableWidget(0, 6)
        self.setups_table.setHorizontalHeaderLabels(
            ["Timestamp", "Close", "Regime", "Direction", "Probability", "Confidence"]
        )

        self.loss_plot = pg.PlotWidget(title="Training Loss")
        self.loss_plot.setLabel("left", "Loss")
        self.loss_plot.setLabel("bottom", "Epoch")

        self.acc_plot = pg.PlotWidget(title="Training Accuracy")
        self.acc_plot.setLabel("left", "Accuracy")
        self.acc_plot.setLabel("bottom", "Epoch")

        metrics_split = QSplitter()
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Regime Distribution"))
        left_layout.addWidget(self.regime_table)
        left_layout.addWidget(QLabel("Confidence Distribution"))
        left_layout.addWidget(self.conf_table)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QLabel("Top High-Confidence Setups (non-synthetic)"))
        right_layout.addWidget(self.setups_table)

        metrics_split.addWidget(left)
        metrics_split.addWidget(right)
        metrics_split.setStretchFactor(0, 1)
        metrics_split.setStretchFactor(1, 2)

        plot_split = QSplitter()
        plot_split.addWidget(self.loss_plot)
        plot_split.addWidget(self.acc_plot)
        plot_split.setStretchFactor(0, 1)
        plot_split.setStretchFactor(1, 1)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addLayout(top)
        layout.addWidget(self.stage_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.summary)
        layout.addWidget(metrics_split, 1)
        layout.addWidget(plot_split, 1)

        self._refresh_summary()

    def set_source_context(self, source_path: str, timeframe_cache: dict):
        self.source_path = source_path
        self.timeframe_cache = timeframe_cache
        self._refresh_summary()

    def set_timeframe_dataset(self, timeframe: str, df):
        self.timeframe_cache[timeframe] = df
        self._refresh_summary()

    def set_dataframe(self, df):
        if df is None:
            return
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
            "AI outputs: regime classes + setup probability/confidence + training curves.",
            "Synthetic policy: synthetic rows are down-ranked and excluded from top setups.",
        ]

        if df is None:
            lines.append("Timeframe status: not loaded yet")
        else:
            lines.append(f"Rows available: {len(df):,}")
            lines.append(f"Columns available: {len(df.columns):,}")

        if self.ai_result is not None:
            lines.append(f"Last avg confidence: {self.ai_result.summary['avg_confidence']:.3f}")
            lines.append(f"Last high-confidence setups: {self.ai_result.summary['high_confidence_rows']:,}")

        self.summary.setPlainText("\n".join(lines))

    def run_ai(self):
        tf = self.timeframe_box.currentText()
        df = self.timeframe_cache.get(tf)

        if df is None:
            QMessageBox.warning(self, "Timeframe not ready", "Selected timeframe is not loaded yet.")
            return

        if self.ai_thread is not None:
            QMessageBox.warning(self, "Already running", "AI analysis is already in progress.")
            return

        self.run_btn.setEnabled(False)
        self.progress.setValue(0)
        self.stage_label.setText("Stage: preparing AI analysis")

        self.regime_table.setRowCount(0)
        self.conf_table.setRowCount(0)
        self.setups_table.setRowCount(0)
        self.loss_plot.clear()
        self.acc_plot.clear()

        self.ai_thread = QThread()
        self.ai_worker = AIWorker(df)
        self.ai_worker.moveToThread(self.ai_thread)

        self.ai_thread.started.connect(self.ai_worker.run)
        self.ai_worker.progress.connect(self.progress.setValue)
        self.ai_worker.stage.connect(lambda t: self.stage_label.setText(f"Stage: {t}"))
        self.ai_worker.log.connect(self.log_message.emit)
        self.ai_worker.finished.connect(self._on_ai_ready)
        self.ai_worker.error.connect(self._on_ai_error)

        self.ai_worker.finished.connect(self.ai_thread.quit)
        self.ai_worker.error.connect(self.ai_thread.quit)
        self.ai_thread.finished.connect(self._cleanup_ai_worker)

        self.ai_thread.start()

    def _on_ai_ready(self, result):
        self.ai_result = result
        self.run_btn.setEnabled(True)
        self.progress.setValue(100)
        self.stage_label.setText("Stage: AI analysis complete")

        self._populate_two_col_table(self.regime_table, result.regime_counts)
        self._populate_two_col_table(self.conf_table, result.confidence_distribution)
        self._populate_setups(result.top_setups)
        self._plot_curves(result.loss_curve, result.accuracy_curve)
        self._refresh_summary()

        self.log_message.emit(
            "INFO",
            f"AI analysis complete | avg_conf={result.summary['avg_confidence']:.3f} | high_conf={result.summary['high_confidence_rows']:,}",
        )

    def _on_ai_error(self, text: str):
        self.run_btn.setEnabled(True)
        self.progress.setValue(0)
        self.stage_label.setText("Stage: AI analysis failed")
        QMessageBox.critical(self, "AI analysis failed", text)

    def _cleanup_ai_worker(self):
        self.ai_worker = None
        self.ai_thread = None

    def _populate_two_col_table(self, table: QTableWidget, data: dict):
        items = list(data.items())
        table.setRowCount(len(items))
        for r, (k, v) in enumerate(items):
            table.setItem(r, 0, QTableWidgetItem(str(k)))
            table.setItem(r, 1, QTableWidgetItem(str(v)))
        table.resizeColumnsToContents()

    def _populate_setups(self, df):
        if df is None or df.empty:
            self.setups_table.setRowCount(0)
            return

        preview = df[["timestamp", "close", "regime", "direction", "setup_probability", "setup_confidence"]]
        preview = preview.head(150).reset_index(drop=True)

        self.setups_table.setRowCount(len(preview))
        for r in range(len(preview)):
            for c, col in enumerate(preview.columns):
                self.setups_table.setItem(r, c, QTableWidgetItem(str(preview.iloc[r, c])))
        self.setups_table.resizeColumnsToContents()

    def _plot_curves(self, loss_curve: list[float], acc_curve: list[float]):
        if loss_curve:
            x = list(range(1, len(loss_curve) + 1))
            self.loss_plot.plot(x, loss_curve, pen=pg.mkPen(color="#ff6b6b", width=2))

        if acc_curve:
            x = list(range(1, len(acc_curve) + 1))
            self.acc_plot.plot(x, acc_curve, pen=pg.mkPen(color="#00d4ff", width=2))
