from __future__ import annotations
from PyQt6.QtCore import pyqtSignal, QThread
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QTextEdit, QHBoxLayout, QMessageBox, QProgressBar, QCheckBox, QLineEdit
from app.core.data_loader import profile_to_text
from app.core.load_worker import LoadWorker
from app.core.cache_manager import dataset_cache_dir

class DataLabPage(QWidget):
    data_loaded = pyqtSignal(object, object)
    log_message = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.df = None
        self.profile = None
        self.source_path = None
        self.worker_thread = None
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        title = QLabel("Data Lab")
        title.setStyleSheet("font-size: 24px; font-weight: 700;")
        subtitle = QLabel("Load refined CSV or Parquet. For huge Parquet files, use a date window so the UI stops hauling unnecessary history.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #8a95a5;")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        row1 = QHBoxLayout()
        self.load_btn = QPushButton("Load CSV / Parquet")
        self.load_btn.clicked.connect(self.load_file)
        row1.addWidget(self.load_btn)
        row1.addStretch(1)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.date_window_check = QCheckBox("Use date window (Parquet only)")
        self.start_edit = QLineEdit()
        self.start_edit.setPlaceholderText("Start UTC, e.g. 2025-01-01")
        self.end_edit = QLineEdit()
        self.end_edit.setPlaceholderText("End UTC, e.g. 2025-03-31")
        row2.addWidget(self.date_window_check)
        row2.addWidget(self.start_edit)
        row2.addWidget(self.end_edit)
        layout.addLayout(row2)

        self.stage_label = QLabel("Stage: idle")
        self.stage_label.setStyleSheet("color: #8a95a5;")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.stage_label)
        layout.addWidget(self.progress)

        self.cache_label = QLabel("Cache: —")
        self.cache_label.setStyleSheet("color: #8a95a5;")
        layout.addWidget(self.cache_label)

        self.profile_box = QTextEdit()
        self.profile_box.setReadOnly(True)
        layout.addWidget(self.profile_box)

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open refined dataset", "", "Data Files (*.parquet *.pq *.csv)")
        if not path:
            return
        self.source_path = path
        self.load_btn.setEnabled(False)
        self.progress.setValue(0)
        self.stage_label.setText("Stage: preparing load")
        self.profile_box.clear()
        self.log_message.emit("INFO", f"User selected file: {path}")

        use_date_window = self.date_window_check.isChecked() and path.lower().endswith((".parquet", ".pq"))
        if self.date_window_check.isChecked() and not path.lower().endswith((".parquet", ".pq")):
            self.log_message.emit("WARN", "Date window loading is only supported for Parquet right now. Falling back to minimal load.")

        self.worker_thread = QThread()
        self.worker = LoadWorker(path, use_date_window=use_date_window, start_text=self.start_edit.text(), end_text=self.end_edit.text())
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.stage.connect(self._set_stage)
        self.worker.log.connect(self.log_message.emit)
        self.worker.finished.connect(self._on_loaded)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.error.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self._cleanup_worker)
        self.worker_thread.start()

    def _set_stage(self, text: str):
        self.stage_label.setText(f"Stage: {text}")

    def _on_loaded(self, df, profile):
        self.df = df
        self.profile = profile
        self.profile_box.setPlainText(profile_to_text(profile))
        self.load_btn.setEnabled(True)
        try:
            self.cache_label.setText(f"Cache: {dataset_cache_dir(self.source_path)}")
        except Exception:
            self.cache_label.setText("Cache: unavailable")
        self.data_loaded.emit(df, profile)

    def _on_error(self, text: str):
        self.load_btn.setEnabled(True)
        self.progress.setValue(0)
        self.stage_label.setText("Stage: load failed")
        QMessageBox.critical(self, "Load failed", text)

    def _cleanup_worker(self):
        self.worker = None
        self.worker_thread = None
