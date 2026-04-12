from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from app.core.ai_engine import analyze_market_ai


class AIWorker(QObject):
    progress = pyqtSignal(int)
    stage = pyqtSignal(str)
    log = pyqtSignal(str, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, df):
        super().__init__()
        self.df = df

    @pyqtSlot()
    def run(self):
        try:
            self.stage.emit("Preparing AI inputs")
            self.progress.emit(10)

            if self.df is None or len(self.df) == 0:
                raise ValueError("AI input dataframe is empty")

            self.log.emit("INFO", f"AI rows: {len(self.df):,}")
            self.progress.emit(35)

            self.stage.emit("Training setup scoring model")
            result = analyze_market_ai(self.df)

            self.progress.emit(90)
            self.stage.emit("Finalizing AI outputs")

            self.log.emit("INFO", f"AI avg confidence: {result.summary['avg_confidence']:.3f}")
            self.log.emit("INFO", f"AI high-confidence setups: {result.summary['high_confidence_rows']:,}")
            self.log.emit("INFO", f"AI final train accuracy: {result.summary['train_final_accuracy']:.3f}")

            self.progress.emit(100)
            self.stage.emit("AI analysis complete")
            self.finished.emit(result)

        except Exception as exc:
            self.progress.emit(0)
            self.stage.emit("AI analysis failed")
            self.log.emit("ERROR", str(exc))
            self.error.emit(str(exc))
