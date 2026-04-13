from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from .backtest_engine import run_backtest, BacktestConfig


class BacktestWorker(QObject):
    progress = pyqtSignal(int)
    stage = pyqtSignal(str)
    log = pyqtSignal(str, str)
    finished = pyqtSignal(object)   # BacktestResult
    error = pyqtSignal(str)

    def __init__(self, df, config: BacktestConfig):
        super().__init__()
        self.df = df
        self.config = config
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _check_cancel(self):
        if self._cancel:
            raise RuntimeError("Backtest cancelled")

    @pyqtSlot()
    def run(self):
        try:
            self.stage.emit("Preparing backtest")
            self.progress.emit(5)

            if self.df is None or len(self.df) == 0:
                raise ValueError("Backtest input dataframe is empty")

            self.log.emit("INFO", f"Backtest rows: {len(self.df):,}")
            self.log.emit("INFO", f"Initial capital: {self.config.initial_capital}")
            self.log.emit("INFO", f"Fee rate: {self.config.fee_rate}")
            self.log.emit("INFO", f"Slippage rate: {self.config.slippage_rate}")

            self._check_cancel()

            self.stage.emit("Running simulation")
            self.progress.emit(40)

            result = run_backtest(self.df, self.config)

            self._check_cancel()

            self.stage.emit("Finalizing results")
            self.progress.emit(85)

            metrics = result.metrics

            self.log.emit("INFO", f"Final equity: {metrics['final_equity']:.2f}")
            self.log.emit("INFO", f"Total return: {metrics['total_return_pct']:.2f}%")
            self.log.emit("INFO", f"Trades: {metrics['total_trades']}")
            self.log.emit("INFO", f"Win rate: {metrics['win_rate_pct']:.2f}%")
            self.log.emit("INFO", f"Max drawdown: {metrics['max_drawdown_pct']:.2f}%")

            self.progress.emit(100)
            self.stage.emit("Backtest complete")

            self.finished.emit(result)

        except Exception as exc:
            self.stage.emit("Backtest failed")
            self.progress.emit(0)
            self.log.emit("ERROR", str(exc))
            self.error.emit(str(exc))