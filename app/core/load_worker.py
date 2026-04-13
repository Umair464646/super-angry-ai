from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from .data_loader import (
    load_market_file_minimal,
    load_parquet_date_window,
    profile_dataframe,
    profile_to_dict,
)
from .cache_manager import write_profile_cache


class LoadWorker(QObject):
    progress = pyqtSignal(int)
    stage = pyqtSignal(str)
    log = pyqtSignal(str, str)
    finished = pyqtSignal(object, object)
    error = pyqtSignal(str)

    def __init__(self, path: str, use_date_window: bool = False, start_text: str = "", end_text: str = ""):
        super().__init__()
        self.path = path
        self.use_date_window = use_date_window
        self.start_text = start_text.strip()
        self.end_text = end_text.strip()

    @pyqtSlot()
    def run(self):
        try:
            self.stage.emit("Loading dataset")
            self.progress.emit(5)
            self.log.emit("INFO", f"Opening file: {self.path}")

            if self.use_date_window:
                self.log.emit(
                    "INFO",
                    f"Load mode: date window | start={self.start_text or '-'} end={self.end_text or '-'}"
                )
                df = load_parquet_date_window(
                    self.path,
                    start=self.start_text or None,
                    end=self.end_text or None,
                )
                profile = profile_dataframe(df, self.path, [])
            else:
                self.log.emit("INFO", "Load mode: minimal chart columns only")
                df, profile = load_market_file_minimal(self.path)

            self.progress.emit(70)
            self.stage.emit("Profiling dataset")
            self.log.emit("INFO", f"Rows loaded: {len(df):,}")
            self.log.emit("INFO", f"Columns loaded: {', '.join(df.columns)}")
            self.log.emit("INFO", f"Date range: {profile.start} -> {profile.end}")
            self.log.emit("INFO", f"Zero-volume bars: {profile.zero_volume_pct:.2f}%")
            self.log.emit("INFO", f"Synthetic rows: {profile.synthetic_pct:.2f}%")

            try:
                write_profile_cache(self.path, profile_to_dict(profile))
                self.log.emit("INFO", "Profile cache updated")
            except Exception as exc:
                self.log.emit("WARN", f"Failed to write profile cache: {exc}")

            self.progress.emit(100)
            self.stage.emit("Load complete")
            self.log.emit("INFO", "Initial dataset load complete")
            self.finished.emit(df, profile)

        except Exception as exc:
            self.log.emit("ERROR", str(exc))
            self.error.emit(str(exc))
