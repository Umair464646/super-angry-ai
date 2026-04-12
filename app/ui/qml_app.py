from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from datetime import datetime, timezone

from PySide6.QtCore import QObject, Property, QTimer, Signal, Slot, QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine


class AppState(QObject):
    strategiesChanged = Signal()
    logsChanged = Signal()
    selectedStrategyChanged = Signal()
    fitnessSeriesChanged = Signal()
    lossSeriesChanged = Signal()
    accuracySeriesChanged = Signal()
    modelStatusChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._strategies: list[dict] = []
        self._logs: list[dict] = []
        self._selected_strategy: dict = {}
        self._fitness_series: list[float] = [0.42, 0.48, 0.54, 0.58]
        self._loss_series: list[float] = [0.92, 0.74, 0.62, 0.51]
        self._accuracy_series: list[float] = [0.43, 0.52, 0.59, 0.66]
        self._model_status = "idle"
        self._generation = 1
        self._candidate = 1

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self.seed_demo_data()

    @Property("QVariantList", notify=strategiesChanged)
    def strategies(self):
        return self._strategies

    @Property("QVariantList", notify=logsChanged)
    def logs(self):
        return self._logs

    @Property("QVariantMap", notify=selectedStrategyChanged)
    def selectedStrategy(self):
        return self._selected_strategy

    @Property("QVariantList", notify=fitnessSeriesChanged)
    def fitnessSeries(self):
        return self._fitness_series

    @Property("QVariantList", notify=lossSeriesChanged)
    def lossSeries(self):
        return self._loss_series

    @Property("QVariantList", notify=accuracySeriesChanged)
    def accuracySeries(self):
        return self._accuracy_series

    @Property(str, notify=modelStatusChanged)
    def modelStatus(self):
        return self._model_status

    @Slot()
    def seed_demo_data(self):
        self._strategies = [
            {
                "id": "GEN1-001",
                "generation": 1,
                "name": "Compression Breakout Mesh",
                "family": "breakout_expansion",
                "origin": "random",
                "fitness": 58.2,
                "robustness": 72.5,
                "validation": 69.7,
                "status": "survived",
                "timeframe": "1m",
                "entry": "EMA21>EMA55 + BB squeeze release + volume spike",
                "exit": "ATR1.6 stop / ATR2.8 target / EMA fail",
            },
            {
                "id": "GEN1-002",
                "generation": 1,
                "name": "Retest Continuation Flow",
                "family": "trend_retest",
                "origin": "random",
                "fitness": 53.4,
                "robustness": 68.1,
                "validation": 64.2,
                "status": "testing",
                "timeframe": "5m",
                "entry": "BOS + retest wick + delta shift",
                "exit": "Structure stop / RR 2.2 / time stop 18 bars",
            },
        ]
        self._logs = [
            {
                "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "level": "INFO",
                "msg": "QML workspace initialized",
            }
        ]
        self._selected_strategy = self._strategies[0]
        self.strategiesChanged.emit()
        self.logsChanged.emit()
        self.selectedStrategyChanged.emit()

    @Slot()
    def startResearch(self):
        self._model_status = "running"
        self.modelStatusChanged.emit()
        self._append_log("INFO", "AI research pipeline started")
        if not self._timer.isActive():
            self._timer.start(1200)

    @Slot()
    def pauseResearch(self):
        self._model_status = "paused"
        self.modelStatusChanged.emit()
        self._append_log("WARN", "Pipeline paused by user")
        self._timer.stop()

    @Slot()
    def stopResearch(self):
        self._model_status = "stopped"
        self.modelStatusChanged.emit()
        self._append_log("INFO", "Pipeline stopped")
        self._timer.stop()

    @Slot(int)
    def selectStrategy(self, index: int):
        if 0 <= index < len(self._strategies):
            self._selected_strategy = self._strategies[index]
            self.selectedStrategyChanged.emit()

    @Slot(str)
    def selectStrategyById(self, strategy_id: str):
        for row in self._strategies:
            if row.get("id") == strategy_id:
                self._selected_strategy = row
                self.selectedStrategyChanged.emit()
                return

    @Slot()
    def copySelectedStrategy(self):
        if not self._selected_strategy:
            return
        block = (
            f"Strategy ID: {self._selected_strategy.get('id')}\n"
            f"Name: {self._selected_strategy.get('name')}\n"
            f"Family: {self._selected_strategy.get('family')}\n"
            f"Generation: {self._selected_strategy.get('generation')}\n"
            f"Timeframe: {self._selected_strategy.get('timeframe')}\n\n"
            f"Entry Logic:\n- {self._selected_strategy.get('entry')}\n\n"
            f"Exit Logic:\n- {self._selected_strategy.get('exit')}\n\n"
            f"Fitness: {self._selected_strategy.get('fitness')}\n"
            f"Robustness: {self._selected_strategy.get('robustness')}\n"
            f"Validation: {self._selected_strategy.get('validation')}\n"
        )
        QGuiApplication.clipboard().setText(block)
        self._append_log("INFO", f"Copied strategy {self._selected_strategy.get('id')} to clipboard")

    @Slot(result=str)
    def selectedStrategyJson(self) -> str:
        return json.dumps(self._selected_strategy, indent=2)

    def _append_log(self, level: str, msg: str):
        self._logs.append({
            "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "level": level,
            "msg": msg,
        })
        self._logs = self._logs[-400:]
        self.logsChanged.emit()

    def _tick(self):
        self._candidate += 1
        if self._candidate % 6 == 0:
            self._generation += 1

        fitness = max(20.0, min(95.0, self._fitness_series[-1] + random.uniform(-1.5, 2.2)))
        loss = max(0.08, self._loss_series[-1] + random.uniform(-0.06, 0.04))
        acc = max(0.25, min(0.99, self._accuracy_series[-1] + random.uniform(-0.02, 0.04)))

        self._fitness_series.append(round(fitness, 3))
        self._loss_series.append(round(loss, 3))
        self._accuracy_series.append(round(acc, 3))
        self._fitness_series = self._fitness_series[-60:]
        self._loss_series = self._loss_series[-60:]
        self._accuracy_series = self._accuracy_series[-60:]
        self.fitnessSeriesChanged.emit()
        self.lossSeriesChanged.emit()
        self.accuracySeriesChanged.emit()

        new_row = {
            "id": f"GEN{self._generation}-{self._candidate:03d}",
            "generation": self._generation,
            "name": random.choice(
                [
                    "Adaptive Flow Mesh",
                    "Volatility Retest Continuation",
                    "Structure Sweep Reversal",
                    "Compression Release Continuation",
                ]
            ),
            "family": random.choice(["trend", "breakout", "reversal", "flow_confirmed"]),
            "origin": random.choice(["mutation", "crossover", "random"]),
            "fitness": round(fitness, 2),
            "robustness": round(random.uniform(50, 88), 2),
            "validation": round(random.uniform(48, 85), 2),
            "status": random.choice(["generated", "backtested", "validated", "survived", "rejected"]),
            "timeframe": random.choice(["1m", "5m", "15m"]),
            "entry": random.choice(
                [
                    "Trend regime + BOS retest + VWAP acceptance",
                    "Range edge rejection + imbalance flip",
                    "Breakout + volume expansion + ADX rising",
                ]
            ),
            "exit": random.choice(
                [
                    "ATR dynamic stop + time stop",
                    "Structure invalidation + adaptive RR",
                    "Trailing ATR + momentum fade",
                ]
            ),
        }
        self._strategies.insert(0, new_row)
        self._strategies = self._strategies[:250]
        self.strategiesChanged.emit()

        if random.random() > 0.55:
            self._selected_strategy = new_row
            self.selectedStrategyChanged.emit()

        self._append_log(
            "AI",
            f"[GEN {self._generation}][CAND {self._candidate:03d}] {new_row['name']} | "
            f"fit={new_row['fitness']:.2f} robust={new_row['robustness']:.2f} status={new_row['status']}",
        )


def run_qml() -> int:
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()

    state = AppState()
    engine.rootContext().setContextProperty("appState", state)
    qml_path = Path(__file__).resolve().parents[1] / "qml" / "Main.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))

    if not engine.rootObjects():
        return 1
    return app.exec()
