from __future__ import annotations

import sys

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QStackedWidget,
    QMessageBox,
    QSplitter,
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt, QThread

from app.core.timeframe_worker import TimeframeWorker
from app.ui.data_lab_page import DataLabPage
from app.ui.chart_lab_page import ChartLabPage
from app.ui.feature_lab_page import FeatureLabPage
from app.ui.strategy_lab_page import StrategyLabPage
from app.ui.backtest_lab_page import BacktestLabPage
from app.ui.validation_lab_page import ValidationLabPage
from app.ui.ai_lab_page import AILabPage
from app.ui.placeholder_page import PlaceholderPage
from app.ui.log_panel import LogPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crypto Strategy Lab V9 Feature Lab")
        self.resize(1640, 980)

        self.base_df = None
        self.profile = None
        self.source_path = None
        self.tf_cache = {}

        self.tf_thread = None
        self.tf_worker = None
        self.tf_target = None

        self._build_ui()
        self._build_menu()
        self._apply_theme()

    def _build_ui(self):
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Vertical)
        root_layout.addWidget(splitter)

        top_container = QWidget()
        top_layout = QHBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)

        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(240)
        self.sidebar.addItems(
            [
                "Projects",
                "Data Lab",
                "Chart Lab",
                "Feature Lab",
                "Strategy Lab",
                "Backtest Lab",
                "Validation Lab",
                "AI Lab",
                "Results",
                "Export",
            ]
        )
        self.sidebar.currentRowChanged.connect(self._switch_page)

        self.stack = QStackedWidget()

        self.projects_page = PlaceholderPage(
            "Projects",
            "Project management is still a placeholder. Load data first.",
        )
        self.data_page = DataLabPage()
        self.chart_page = ChartLabPage()
        self.feature_page = FeatureLabPage()
        self.strategy_page = StrategyLabPage()
        self.backtest_page = BacktestLabPage()
        self.validation_page = ValidationLabPage()
        self.ai_page = AILabPage()
        self.results_page = PlaceholderPage(
            "Results",
            "Ranking, robustness views, and exports come later.",
        )
        self.export_page = PlaceholderPage(
            "Export",
            "Rule sheets, JSON configs, and trade logs come later.",
        )

        for page in [
            self.projects_page,
            self.data_page,
            self.chart_page,
            self.feature_page,
            self.strategy_page,
            self.backtest_page,
            self.validation_page,
            self.ai_page,
            self.results_page,
            self.export_page,
        ]:
            self.stack.addWidget(page)

        top_layout.addWidget(self.sidebar)
        top_layout.addWidget(self.stack, 1)

        self.log_panel = LogPanel()

        splitter.addWidget(top_container)
        splitter.addWidget(self.log_panel)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([760, 220])

        self.setCentralWidget(root)
        self.sidebar.setCurrentRow(1)

        self.data_page.data_loaded.connect(self.on_data_loaded)
        self.data_page.log_message.connect(self.log_panel.append)

        self.chart_page.timeframe_requested.connect(self.build_timeframe_async)
        self.feature_page.timeframe_requested.connect(self.build_timeframe_async)
        self.feature_page.log_message.connect(self.log_panel.append)

        self.strategy_page.timeframe_requested.connect(self.build_timeframe_async)
        self.strategy_page.log_message.connect(self.log_panel.append)

        self.backtest_page.timeframe_requested.connect(self.build_timeframe_async)
        self.backtest_page.log_message.connect(self.log_panel.append)

        self.validation_page.timeframe_requested.connect(self.build_timeframe_async)
        self.validation_page.log_message.connect(self.log_panel.append)

        self.ai_page.timeframe_requested.connect(self.build_timeframe_async)
        self.ai_page.log_message.connect(self.log_panel.append)

    def _build_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        open_action = QAction("Open dataset", self)
        open_action.triggered.connect(self.data_page.load_file)
        file_menu.addAction(open_action)

        help_menu = menubar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def _switch_page(self, idx: int):
        self.stack.setCurrentIndex(max(idx, 0))

    def on_data_loaded(self, df, profile):
        self.base_df = df
        self.profile = profile
        self.source_path = profile.path
        self.tf_cache = {"1s": df}
        self.tf_target = None

        self.chart_page.set_base_dataset(df)
        self.feature_page.set_source_context(self.source_path, self.tf_cache)
        self.strategy_page.set_source_context(self.source_path, self.tf_cache)
        self.backtest_page.set_source_context(self.source_path, self.tf_cache)
        self.validation_page.set_source_context(self.source_path, self.tf_cache)
        self.ai_page.set_source_context(self.source_path, self.tf_cache)
        self.ai_page.set_dataframe(df)

        self.log_panel.append(
            "INFO",
            "Base dataset propagated to Chart/Feature/Strategy/Backtest/Validation/AI labs",
        )
        self.sidebar.setCurrentRow(2)

    def build_timeframe_async(self, timeframe: str):
        if self.base_df is None or self.source_path is None:
            self.log_panel.append("WARN", "Cannot build timeframe before loading data")
            return

        if timeframe in self.tf_cache:
            self.log_panel.append("INFO", f"Timeframe already cached in memory: {timeframe}")
            self.chart_page.set_timeframe_dataset(timeframe, self.tf_cache[timeframe])
            self.feature_page.set_timeframe_dataset(timeframe, self.tf_cache[timeframe])
            self.strategy_page.set_timeframe_dataset(timeframe, self.tf_cache[timeframe])
            self.backtest_page.set_timeframe_dataset(timeframe, self.tf_cache[timeframe])
            self.validation_page.set_timeframe_dataset(timeframe, self.tf_cache[timeframe])
            self.ai_page.set_timeframe_dataset(timeframe, self.tf_cache[timeframe])
            return

        if self.tf_thread is not None:
            if timeframe == self.tf_target:
                self.log_panel.append(
                    "WARN",
                    f"Timeframe build already in progress for {timeframe}",
                )
            else:
                self.log_panel.append(
                    "WARN",
                    f"Already building {self.tf_target}. Wait for it to finish before requesting {timeframe}.",
                )
            return

        self.tf_target = timeframe
        self.log_panel.append("INFO", f"Requested timeframe build: {timeframe}")

        self.data_page.progress.setValue(0)
        self.data_page.stage_label.setText(f"Stage: preparing timeframe {timeframe}")
        self.data_page.load_btn.setEnabled(False)

        self.tf_thread = QThread()
        self.tf_worker = TimeframeWorker(self.base_df, self.source_path, timeframe)
        self.tf_worker.moveToThread(self.tf_thread)

        self.tf_thread.started.connect(self.tf_worker.run)

        self.tf_worker.log.connect(self.log_panel.append)
        self.tf_worker.stage.connect(self._on_tf_stage)
        self.tf_worker.progress.connect(self._on_tf_progress)
        self.tf_worker.finished.connect(self._on_timeframe_ready)
        self.tf_worker.error.connect(self._on_timeframe_error)

        self.tf_worker.finished.connect(self.tf_thread.quit)
        self.tf_worker.error.connect(self.tf_thread.quit)
        self.tf_thread.finished.connect(self._cleanup_tf_worker)

        self.tf_thread.start()

    def _on_tf_stage(self, text: str):
        if self.tf_target:
            self.data_page.stage_label.setText(f"Stage: {text} [{self.tf_target}]")
        else:
            self.data_page.stage_label.setText(f"Stage: {text}")

    def _on_tf_progress(self, value: int):
        self.data_page.progress.setValue(max(0, min(100, int(value))))

    def _on_timeframe_ready(self, timeframe, df):
        self.tf_cache[timeframe] = df
        self.chart_page.set_timeframe_dataset(timeframe, df)
        self.feature_page.set_timeframe_dataset(timeframe, df)
        self.strategy_page.set_timeframe_dataset(timeframe, df)
        self.backtest_page.set_timeframe_dataset(timeframe, df)
        self.validation_page.set_timeframe_dataset(timeframe, df)
        self.ai_page.set_timeframe_dataset(timeframe, df)

        self.data_page.progress.setValue(100)
        self.data_page.stage_label.setText(f"Stage: timeframe ready [{timeframe}]")
        self.data_page.load_btn.setEnabled(True)

        self.log_panel.append(
            "INFO",
            f"Timeframe propagated to Chart/Feature/Strategy/Backtest/Validation labs: {timeframe} ({len(df):,} rows)",
        )

    def _on_timeframe_error(self, text: str):
        self.data_page.progress.setValue(0)
        self.data_page.stage_label.setText("Stage: timeframe build failed")
        self.data_page.load_btn.setEnabled(True)
        QMessageBox.critical(self, "Timeframe build failed", text)

    def _cleanup_tf_worker(self):
        self.tf_worker = None
        self.tf_thread = None
        self.tf_target = None

    def show_about(self):
        QMessageBox.information(
            self,
            "About Crypto Strategy Lab V9 Feature Lab",
            "Build with Data/Chart/Feature/Strategy/Backtest/Validation labs and timeframe caching.",
        )

    def _apply_theme(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #0a0c10;
                color: #e6edf3;
                font-family: Segoe UI;
                font-size: 13px;
            }
            QMenuBar, QMenu {
                background: #0f1218;
                color: #e6edf3;
            }
            QListWidget {
                background: #0f1218;
                border: none;
                color: #93a1b3;
                font-size: 13px;
            }
            QListWidget::item {
                padding: 12px 14px;
                border-left: 3px solid transparent;
            }
            QListWidget::item:selected {
                background: #001a22;
                color: #00d4ff;
                border-left: 3px solid #00d4ff;
            }
            QPushButton {
                background: #00d4ff;
                color: #000;
                padding: 8px 14px;
                border: none;
                border-radius: 4px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #00b7dd;
            }
            QPushButton:disabled {
                background: #24313d;
                color: #76879a;
            }
            QTextEdit, QComboBox, QPlainTextEdit, QLineEdit, QTableWidget {
                background: #0f1218;
                border: 1px solid #1f2933;
                color: #e6edf3;
            }
            QHeaderView::section {
                background: #131720;
                color: #9fb0c0;
                border: 1px solid #1f2933;
                padding: 6px;
            }
            QLabel, QCheckBox {
                color: #e6edf3;
            }
            QProgressBar {
                border: 1px solid #1f2933;
                background: #0f1218;
                color: #e6edf3;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #00d4ff;
            }
            QSplitter::handle {
                background: #11161d;
                height: 6px;
            }
            """
        )


def run():
    app = QApplication(sys.argv)
    app.setApplicationName("Crypto Strategy Lab V9 Feature Lab")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
