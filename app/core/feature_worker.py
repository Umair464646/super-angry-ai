from __future__ import annotations
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from .feature_engine import generate_features

class FeatureWorker(QObject):
    progress = pyqtSignal(int)
    stage = pyqtSignal(str)
    log = pyqtSignal(str, str)
    finished = pyqtSignal(object, object)
    error = pyqtSignal(str)

    def __init__(self, df, selected_features: list[str]):
        super().__init__()
        self.df = df
        self.selected_features = selected_features

    @pyqtSlot()
    def run(self):
        try:
            self.stage.emit("Preparing feature generation")
            self.progress.emit(5)
            self.log.emit("INFO", f"Selected feature groups: {', '.join(self.selected_features)}")
            self.log.emit("INFO", f"Input rows: {len(self.df):,}")

            self.stage.emit("Generating features")
            self.progress.emit(30)
            out, cols = generate_features(self.df, self.selected_features)

            self.stage.emit("Finalizing output")
            self.progress.emit(100)
            self.log.emit("INFO", f"Generated columns: {len(cols)}")
            self.log.emit("INFO", f"Output columns: {len(out.columns)}")
            self.finished.emit(out, cols)
        except Exception as exc:
            self.log.emit("ERROR", str(exc))
            self.error.emit(str(exc))
