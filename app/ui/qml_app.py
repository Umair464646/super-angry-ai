from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QObject, Property, QThread, Signal, Slot, QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

from app.core.ai_engine import analyze_market_ai
from app.core.data_loader import load_market_file_minimal
from app.core.feature_engine import generate_features
from app.core.strategy_engine import evolve_templates, walk_forward_validate


class ResearchWorker(QObject):
    log = Signal(str, str)
    strategy = Signal(object)
    stage = Signal(str)
    ai_epoch = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, dataset_path: str, generations: int, population_top_k: int, model_type: str):
        super().__init__()
        self.dataset_path = dataset_path
        self.generations = generations
        self.population_top_k = population_top_k
        self.model_type = model_type
        self._cancel = False

    @Slot()
    def cancel(self):
        self._cancel = True

    @Slot()
    def run(self):
        try:
            if not self.dataset_path:
                raise ValueError("No dataset selected. Enter a CSV/Parquet path in the top bar.")
            if not Path(self.dataset_path).exists():
                raise ValueError(f"Dataset not found: {self.dataset_path}")

            self.stage.emit("Loading dataset")
            df, profile = load_market_file_minimal(self.dataset_path)
            self.log.emit("INFO", f"Loaded dataset rows={len(df):,} synthetic={profile.synthetic_pct:.2f}%")
            if self._cancel:
                return

            self.stage.emit("Feature engineering")
            selected_features = [
                "EMA", "RSI", "MACD", "ATR", "BOLLINGER", "VWAP", "MOMENTUM", "ORDER_FLOW", "BREAKOUT",
            ]
            featured_df, generated_cols = generate_features(df, selected_features)
            self.log.emit("INFO", f"Generated {len(generated_cols)} features")
            if self._cancel:
                return

            working_df = featured_df
            if len(working_df) > 10000:
                stride = max(1, len(working_df) // 10000)
                working_df = working_df.iloc[::stride].reset_index(drop=True)
                self.log.emit("WARN", f"Downsampled research rows for responsiveness: {len(featured_df):,}->{len(working_df):,}")

            self.stage.emit("Strategy evolution")
            generation_fitness: list[float] = []
            seed_pool: list[dict] = []
            last_top = None
            strategy_counter = 0

            for gen in range(1, self.generations + 1):
                if self._cancel:
                    return
                all_variants, top_variants = evolve_templates(
                    working_df,
                    top_k=self.population_top_k,
                    seed_pool=seed_pool,
                    max_variants=180,
                )
                if top_variants.empty:
                    raise RuntimeError("No strategy variants generated")

                best = top_variants.iloc[0]
                generation_fitness.append(float(best["fitness"]))
                last_top = best

                survivors = set(
                    (str(r["template_key"]), str(sorted(dict(r["params"]).items())))
                    for _, r in top_variants.iterrows()
                )
                for _, row in all_variants.iterrows():
                    strategy_counter += 1
                    key_sig = (str(row["template_key"]), str(sorted(dict(row["params"]).items())))
                    payload = {
                        "id": f"GEN{gen}-{strategy_counter:04d}",
                        "generation": gen,
                        "name": str(row["strategy"]),
                        "family": str(row["template_key"]),
                        "origin": str(row.get("origin", "random")),
                        "fitness": round(float(row["fitness"]), 4),
                        "robustness": round(float(row["robustness_score"]), 4),
                        "validation": round(float(row["robustness_score"]), 4),
                        "status": "survived" if key_sig in survivors else "rejected",
                        "timeframe": "active",
                        "entry": str(row.get("entry_logic", "deterministic rules")),
                        "exit": str(row.get("exit_logic", "risk model")),
                        "params": dict(row["params"]),
                    }
                    self.strategy.emit(payload)

                seed_pool = top_variants[["template_key", "params"]].to_dict("records")
                self.log.emit("INFO", f"Generation {gen}/{self.generations} best fitness={float(best['fitness']):.2f}")

            if last_top is None:
                raise RuntimeError("Evolution completed with no best strategy")

            self.stage.emit("Walk-forward validation")
            wf, stability = walk_forward_validate(
                working_df,
                template_key=str(last_top["template_key"]),
                params=dict(last_top["params"]),
                folds=4,
            )
            self.log.emit("INFO", f"Walk-forward stability={stability:.2f}")

            self.stage.emit("AI analysis")

            def _epoch_cb(epoch, total, loss, acc, extra):
                payload = {
                    "epoch": int(epoch),
                    "total": int(total),
                    "loss": float(loss),
                    "acc": float(acc),
                    "val_loss": float(extra.get("val_loss", loss)),
                    "val_acc": float(extra.get("val_acc", acc)),
                    "prediction": str(extra.get("prediction", "n/a")),
                    "probability": float(extra.get("probability", 0.0)),
                    "confidence": float(extra.get("output_confidence", 0.0)),
                }
                self.ai_epoch.emit(payload)

            ai = analyze_market_ai(working_df, model_type=self.model_type, epoch_cb=_epoch_cb)

            payload = {
                "profile": asdict(profile),
                "feature_count": len(generated_cols),
                "fitness_series": generation_fitness,
                "wf_rows": wf.to_dict("records"),
                "stability": float(stability),
                "ai": {
                    "loss": ai.loss_curve,
                    "accuracy": ai.accuracy_curve,
                    "val_loss": ai.val_loss_curve,
                    "val_accuracy": ai.val_accuracy_curve,
                    "regime_counts": ai.regime_counts,
                    "confidence_distribution": ai.confidence_distribution,
                    "prediction_distribution": ai.prediction_distribution,
                    "feature_importance": ai.feature_importance,
                    "summary": ai.summary,
                    "model_notes": ai.model_notes,
                    "architecture": ai.nn_architecture,
                },
            }
            self.finished.emit(payload)
        except Exception as exc:
            self.failed.emit(str(exc))


class AppState(QObject):
    strategiesChanged = Signal()
    logsChanged = Signal()
    selectedStrategyChanged = Signal()
    fitnessSeriesChanged = Signal()
    lossSeriesChanged = Signal()
    accuracySeriesChanged = Signal()
    valLossSeriesChanged = Signal()
    valAccuracySeriesChanged = Signal()
    modelStatusChanged = Signal()
    datasetPathChanged = Signal()
    stageTextChanged = Signal()
    regimeCountsChanged = Signal()
    featureImportanceChanged = Signal()
    profileChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._strategies: list[dict] = []
        self._logs: list[dict] = []
        self._selected_strategy: dict = {}
        self._fitness_series: list[float] = []
        self._loss_series: list[float] = []
        self._accuracy_series: list[float] = []
        self._val_loss_series: list[float] = []
        self._val_accuracy_series: list[float] = []
        self._regime_counts: dict = {}
        self._feature_importance: dict = {}
        self._model_status = "idle"
        self._dataset_path = ""
        self._stage_text = "Idle"
        self._profile: dict = {}
        self._base_df: pd.DataFrame | None = None

        self._thread: QThread | None = None
        self._worker: ResearchWorker | None = None

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

    @Property("QVariantList", notify=valLossSeriesChanged)
    def valLossSeries(self):
        return self._val_loss_series

    @Property("QVariantList", notify=valAccuracySeriesChanged)
    def valAccuracySeries(self):
        return self._val_accuracy_series

    @Property("QVariantMap", notify=regimeCountsChanged)
    def regimeCounts(self):
        return self._regime_counts

    @Property("QVariantMap", notify=featureImportanceChanged)
    def featureImportance(self):
        return self._feature_importance


    @Property("QVariantMap", notify=profileChanged)
    def profile(self):
        return self._profile

    @Property(str, notify=modelStatusChanged)
    def modelStatus(self):
        return self._model_status

    @Property(str, notify=datasetPathChanged)
    def datasetPath(self):
        return self._dataset_path

    @Property(str, notify=stageTextChanged)
    def stageText(self):
        return self._stage_text

    @Slot(str)
    def setDatasetPath(self, dataset_path: str):
        self._dataset_path = dataset_path.strip()
        self.datasetPathChanged.emit()


    @Slot()
    def loadDataset(self):
        try:
            if not self._dataset_path:
                raise ValueError("No dataset selected")
            df, profile = load_market_file_minimal(self._dataset_path)
            self._base_df = df
            self._profile = asdict(profile)
            self.profileChanged.emit()
            self._set_stage("Dataset loaded")
            self._append_log("INFO", f"Dataset loaded: rows={profile.rows:,} cols={len(profile.columns)}")
            self._append_log("INFO", f"Range: {profile.start} -> {profile.end}")
        except Exception as exc:
            self._append_log("ERROR", f"Dataset load failed: {exc}")

    @Slot()
    def startResearch(self):
        if self._thread is not None:
            self._append_log("WARN", "Research already running")
            return
        self._strategies = []
        self._selected_strategy = {}
        self._fitness_series = []
        self._loss_series = []
        self._accuracy_series = []
        self._val_loss_series = []
        self._val_accuracy_series = []
        self._regime_counts = {}
        self._feature_importance = {}
        self._profile = {}
        self.strategiesChanged.emit()
        self.selectedStrategyChanged.emit()
        self.fitnessSeriesChanged.emit()
        self.lossSeriesChanged.emit()
        self.accuracySeriesChanged.emit()
        self.valLossSeriesChanged.emit()
        self.valAccuracySeriesChanged.emit()
        self.profileChanged.emit()

        if self._base_df is None:
            self.loadDataset()
            if self._base_df is None:
                return

        self._model_status = "running"
        self.modelStatusChanged.emit()
        self._set_stage("Preparing")

        self._thread = QThread()
        self._worker = ResearchWorker(self._dataset_path, generations=4, population_top_k=10, model_type="mlp")
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append_log)
        self._worker.stage.connect(self._set_stage)
        self._worker.strategy.connect(self._on_strategy)
        self._worker.ai_epoch.connect(self._on_ai_epoch)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)

        self._thread.start()

    @Slot()
    def pauseResearch(self):
        self._append_log("WARN", "Pause is not implemented for QML orchestrator yet")

    @Slot()
    def stopResearch(self):
        if self._worker is not None:
            self._worker.cancel()
        self._append_log("INFO", "Cancellation requested")

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
            f"Parameters: {json.dumps(self._selected_strategy.get('params', {}), indent=2)}\n\n"
            f"Fitness: {self._selected_strategy.get('fitness')}\n"
            f"Robustness: {self._selected_strategy.get('robustness')}\n"
            f"Validation: {self._selected_strategy.get('validation')}\n"
        )
        QGuiApplication.clipboard().setText(block)
        self._append_log("INFO", f"Copied strategy {self._selected_strategy.get('id')} to clipboard")

    @Slot(result=str)
    def selectedStrategyJson(self) -> str:
        return json.dumps(self._selected_strategy, indent=2)

    @Slot(str, str)
    def _append_log(self, level: str, msg: str):
        from datetime import datetime, timezone

        self._logs.append({
            "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "level": level,
            "msg": msg,
        })
        self._logs = self._logs[-600:]
        self.logsChanged.emit()

    @Slot(str)
    def _set_stage(self, stage: str):
        self._stage_text = stage
        self.stageTextChanged.emit()

    @Slot(object)
    def _on_strategy(self, payload: object):
        row = dict(payload)
        self._strategies.insert(0, row)
        self._strategies = self._strategies[:800]
        self.strategiesChanged.emit()
        if not self._selected_strategy:
            self._selected_strategy = row
            self.selectedStrategyChanged.emit()

    @Slot(object)
    def _on_ai_epoch(self, payload: object):
        p = dict(payload)
        self._loss_series.append(float(p.get("loss", 0.0)))
        self._accuracy_series.append(float(p.get("acc", 0.0)))
        self._val_loss_series.append(float(p.get("val_loss", p.get("loss", 0.0))))
        self._val_accuracy_series.append(float(p.get("val_acc", p.get("acc", 0.0))))
        self.lossSeriesChanged.emit()
        self.accuracySeriesChanged.emit()
        self.valLossSeriesChanged.emit()
        self.valAccuracySeriesChanged.emit()

    @Slot(object)
    def _on_finished(self, payload: object):
        data = dict(payload)
        self._model_status = "completed"
        self.modelStatusChanged.emit()
        self._set_stage("Completed")

        self._profile = dict(data.get("profile", {}))
        self.profileChanged.emit()

        self._fitness_series = list(data.get("fitness_series", []))
        self.fitnessSeriesChanged.emit()

        ai = dict(data.get("ai", {}))
        self._regime_counts = dict(ai.get("regime_counts", {}))
        self._feature_importance = dict(ai.get("feature_importance", {}))
        self.regimeCountsChanged.emit()
        self.featureImportanceChanged.emit()

        self._append_log("INFO", "Research pipeline completed")
        self._append_log("INFO", f"Stability score={float(data.get('stability', 0.0)):.2f}")

        self._cleanup_worker()

    @Slot(str)
    def _on_failed(self, message: str):
        self._model_status = "error"
        self.modelStatusChanged.emit()
        self._set_stage("Failed")
        self._append_log("ERROR", message)
        self._cleanup_worker()

    def _cleanup_worker(self):
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None


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
