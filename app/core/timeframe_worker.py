from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
import pandas as pd

from .resampler import build_timeframe
from .cache_manager import timeframe_cache_path


class TimeframeWorker(QObject):
    log = pyqtSignal(str, str)
    stage = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal(str, object)
    error = pyqtSignal(str)

    def __init__(self, df, source_path: str, timeframe: str):
        super().__init__()
        self.df = df
        self.source_path = source_path
        self.timeframe = timeframe
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def _check_cancelled(self):
        if self._cancel_requested:
            raise RuntimeError(f"Timeframe build cancelled: {self.timeframe}")

    @pyqtSlot()
    def run(self):
        try:
            self.stage.emit("Preparing timeframe build")
            self.progress.emit(5)
            self.log.emit("INFO", f"Timeframe request started: {self.timeframe}")
            self._check_cancelled()

            if self.df is None or len(self.df) == 0:
                raise ValueError("Base dataframe is empty. Load data before building timeframes.")

            cache_path = timeframe_cache_path(self.source_path, self.timeframe)
            self.log.emit("INFO", f"Resolved timeframe cache path: {cache_path}")
            self.progress.emit(12)
            self._check_cancelled()

            if cache_path.exists():
                self.stage.emit("Loading timeframe from cache")
                self.log.emit("INFO", f"Cache hit for {self.timeframe}: {cache_path.name}")
                self.progress.emit(30)
                out = pd.read_parquet(cache_path)
                self._check_cancelled()

                if "timestamp" in out.columns:
                    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
                    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

                self.progress.emit(100)
                self.stage.emit("Timeframe ready")
                self.log.emit(
                    "INFO",
                    f"Loaded cached timeframe {self.timeframe}: {len(out):,} rows"
                )
                self.finished.emit(self.timeframe, out)
                return

            self.stage.emit("Building timeframe")
            self.log.emit(
                "INFO",
                f"Cache miss for {self.timeframe}. Building from base dataframe with {len(self.df):,} rows"
            )
            self.progress.emit(35)
            self._check_cancelled()

            out = build_timeframe(self.df, self.timeframe)
            self._check_cancelled()

            self.progress.emit(78)
            self.stage.emit("Saving timeframe cache")
            try:
                out.to_parquet(cache_path, index=False)
                self.log.emit("INFO", f"Saved timeframe cache: {cache_path.name}")
            except Exception as exc:
                self.log.emit("WARN", f"Failed to save timeframe cache for {self.timeframe}: {exc}")

            self._check_cancelled()

            if "timestamp" in out.columns:
                out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
                out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

            self.progress.emit(100)
            self.stage.emit("Timeframe ready")

            if len(out) > 0:
                start_ts = str(out["timestamp"].iloc[0]) if "timestamp" in out.columns else "-"
                end_ts = str(out["timestamp"].iloc[-1]) if "timestamp" in out.columns else "-"
                self.log.emit(
                    "INFO",
                    f"Built {self.timeframe}: {len(out):,} rows | range {start_ts} -> {end_ts}"
                )
            else:
                self.log.emit("WARN", f"Built {self.timeframe}, but result is empty")

            self.finished.emit(self.timeframe, out)

        except Exception as exc:
            self.stage.emit("Timeframe failed")
            self.progress.emit(0)
            self.log.emit("ERROR", f"Timeframe build failed for {self.timeframe}: {exc}")
            self.error.emit(str(exc))