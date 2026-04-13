from __future__ import annotations

import json
import hashlib
import sys
from urllib.parse import unquote, urlparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import numpy as np
from PySide6.QtCore import QObject, Property, QThread, Signal, Slot, QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

from app.core.ai_engine import analyze_market_ai
from app.core.data_loader import load_market_file_minimal
from app.core.feature_engine import generate_features
from app.core.strategy_engine import evolve_templates, walk_forward_validate, TEMPLATES
from app.core.chart_adapter import build_candle_payload


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

    def _template_details(self, template_key: str):
        for template in TEMPLATES:
            if template.key == template_key:
                return template
        return None

    def _regime_hint(self, template_key: str, indicators: list[str], context: dict | None = None) -> str:
        key = (template_key or "").lower()
        inds = [str(i).lower() for i in indicators]
        observed_tr = float((context or {}).get("ctx_trending_avg_return", 0.0) or 0.0)
        observed_rg = float((context or {}).get("ctx_ranging_avg_return", 0.0) or 0.0)
        observed_hv = float((context or {}).get("ctx_high_vol_avg_return", 0.0) or 0.0)
        observed_lv = float((context or {}).get("ctx_low_vol_avg_return", 0.0) or 0.0)
        trend_conf = float((context or {}).get("ctx_trend_confidence", 0.0) or 0.0)
        vol_conf = float((context or {}).get("ctx_volatility_confidence", 0.0) or 0.0)
        sample_count = max(1.0, float((context or {}).get("ctx_sample_count", 0.0) or 1.0))
        return_scale = max(1e-9, float((context or {}).get("ctx_return_scale", 0.0) or 1e-9))

        template_regime = "trend-following"
        if "breakout" in key or "breakout" in inds or "donchian" in inds:
            template_regime = "breakout"
        elif "reversal" in key or "mean" in key or "rsi" in inds or "zscore" in inds:
            template_regime = "mean-reversion"

        min_required_conf = 1.0 / (1.0 + np.sqrt(sample_count))
        adaptive_gap = return_scale / np.sqrt(sample_count)
        if max(trend_conf, vol_conf) < min_required_conf:
            return "uncertain"
        if observed_tr - observed_rg > adaptive_gap:
            return "trend-following"
        if observed_rg - observed_tr > adaptive_gap:
            return "mean-reversion"
        if observed_hv - observed_lv > adaptive_gap and template_regime in {"breakout", "trend-following"}:
            return "breakout"
        return template_regime

    def _strategy_explanation(self, template_key: str, params: dict, context: dict | None = None) -> dict:
        template = self._template_details(template_key)
        if template is None:
            return {
                "indicators": "n/a",
                "entry": "n/a",
                "exit": "n/a",
                "summary": "No template metadata available.",
                "regime": "trend-following",
            }
        indicator_text = ", ".join([str(i) for i in template.indicators]) or "n/a"
        entry_text = str(template.entry_logic)
        exit_text = str(template.exit_logic)
        param_text = ", ".join(f"{k}={v}" for k, v in sorted(dict(params).items()))
        regime = self._regime_hint(template_key, list(template.indicators), context=context)
        summary = f"Uses {indicator_text}. Entry: {entry_text}. Exit: {exit_text}. Params: {param_text}."
        return {
            "indicators": indicator_text,
            "entry": entry_text,
            "exit": exit_text,
            "summary": summary,
            "regime": regime,
        }

    @Slot()
    def run(self):
        try:
            if not self.dataset_path:
                raise ValueError("No dataset selected. Enter a CSV/Parquet path in the top bar.")
            if not Path(self.dataset_path).exists():
                raise ValueError(f"Dataset not found: {self.dataset_path}")

            self.stage.emit("Dataset load")
            self.log.emit("INFO", "dataset load started")
            df, profile = load_market_file_minimal(self.dataset_path)
            self.log.emit("INFO", f"dataset ready: rows={len(df):,} synthetic={profile.synthetic_pct:.2f}%")
            if self._cancel:
                return

            self.stage.emit("Feature generation")
            self.log.emit("INFO", "feature generation started")
            selected_features = [
                "EMA", "RSI", "MACD", "ATR", "BOLLINGER", "VWAP", "MOMENTUM", "ORDER_FLOW", "BREAKOUT",
            ]
            featured_df, generated_cols = generate_features(df, selected_features)
            self.log.emit("INFO", f"feature generation completed: generated={len(generated_cols)}")
            if self._cancel:
                return

            working_df = featured_df
            if len(working_df) > 10000:
                stride = max(1, len(working_df) // 10000)
                working_df = working_df.iloc[::stride].reset_index(drop=True)
                self.log.emit("WARN", f"Downsampled research rows for responsiveness: {len(featured_df):,}->{len(working_df):,}")

            self.stage.emit("Strategy generation")
            self.log.emit("INFO", "strategy generation started")
            generation_fitness: list[float] = []
            seed_pool: list[dict] = []
            last_top = None
            strategy_counter = 0

            for gen in range(1, self.generations + 1):
                if self._cancel:
                    return
                self.stage.emit(f"Backtest generation {gen}/{self.generations}")
                self.log.emit("INFO", f"backtest started: generation={gen}")
                strategy_ids: dict[str, str] = {}

                def _row_to_id(row: dict) -> str:
                    sig = f"{row.get('template_key','')}|{json.dumps(dict(row.get('params', {})), sort_keys=True)}"
                    digest = hashlib.md5(sig.encode("utf-8")).hexdigest()[:10]
                    sid = f"G{gen}-{digest}"
                    strategy_ids[sig] = sid
                    return sid

                def _emit_streamed_variant(done: int, total: int, row: dict):
                    sid = _row_to_id(row)
                    context = {
                        "ctx_high_vol_avg_return": row.get("ctx_high_vol_avg_return", 0.0),
                        "ctx_low_vol_avg_return": row.get("ctx_low_vol_avg_return", 0.0),
                        "ctx_trending_avg_return": row.get("ctx_trending_avg_return", 0.0),
                        "ctx_ranging_avg_return": row.get("ctx_ranging_avg_return", 0.0),
                        "ctx_trend_confidence": row.get("ctx_trend_confidence", 0.0),
                        "ctx_volatility_confidence": row.get("ctx_volatility_confidence", 0.0),
                        "ctx_sample_count": row.get("ctx_sample_count", 0),
                        "ctx_return_scale": row.get("ctx_return_scale", 0.0),
                    }
                    explanation = self._strategy_explanation(
                        str(row.get("template_key", "")),
                        dict(row.get("params", {})),
                        context=context,
                    )
                    payload = {
                        "id": sid,
                        "generation": gen,
                        "name": str(row.get("strategy", "n/a")),
                        "family": str(row.get("template_key", "n/a")),
                        "origin": str(row.get("origin", "random")),
                        "fitness": round(float(row.get("fitness", 0.0)), 4),
                        "robustness": round(float(row.get("robustness_score", 0.0)), 4),
                        "validation": "pending",
                        "status": "backtested",
                        "timeframe": "active",
                        "entry": explanation["entry"],
                        "exit": explanation["exit"],
                        "indicators": explanation["indicators"],
                        "explanation": explanation["summary"],
                        "regime": explanation["regime"],
                        "params": dict(row.get("params", {})),
                        "trade_count": int(row.get("test_trades", 0)),
                        "win_rate": round(float(row.get("test_win_rate_pct", 0.0)), 2),
                        "pnl": round(float(row.get("test_return_pct", 0.0)), 4),
                        "drawdown": round(float(row.get("test_max_drawdown_pct", 0.0)), 4),
                        "avg_trade_return": round(float(row.get("test_avg_trade_return_pct", 0.0)), 4),
                        "max_win": round(float(row.get("test_max_win_pct", 0.0)), 4),
                        "max_loss": round(float(row.get("test_max_loss_pct", 0.0)), 4),
                        "trade_distribution": f"W{int(row.get('test_win_trades', 0))}/L{int(row.get('test_loss_trades', 0))}",
                        "performance_context": str(row.get("performance_context", "")),
                        "ctx_high_vol_avg_return": round(float(row.get("ctx_high_vol_avg_return", 0.0)), 4),
                        "ctx_low_vol_avg_return": round(float(row.get("ctx_low_vol_avg_return", 0.0)), 4),
                        "ctx_trending_avg_return": round(float(row.get("ctx_trending_avg_return", 0.0)), 4),
                        "ctx_ranging_avg_return": round(float(row.get("ctx_ranging_avg_return", 0.0)), 4),
                        "context_confidence": round(float(row.get("ctx_confidence", 0.0)), 4),
                        "trend_context_confidence": round(float(row.get("ctx_trend_confidence", 0.0)), 4),
                        "volatility_context_confidence": round(float(row.get("ctx_volatility_confidence", 0.0)), 4),
                        "behavior_robustness": round(float(row.get("behavior_robustness", 0.0)), 2),
                        "time_stability": round(float(row.get("ctx_time_stability", 0.0)), 4),
                        "decay_score": round(float(row.get("ctx_decay_score", 0.0)), 4),
                        "decay_flag": bool(row.get("ctx_decay_flag", False)),
                        "ctx_return_scale": round(float(row.get("ctx_return_scale", 0.0)), 6),
                    }
                    self.strategy.emit(payload)
                    self.log.emit(
                        "INFO",
                        f"backtest update [{done}/{total}] {payload['id']} trades={payload['trade_count']} "
                        f"win={payload['win_rate']:.2f}% pnl={payload['pnl']:.2f}% dd={payload['drawdown']:.2f}%"
                    )

                all_variants, top_variants = evolve_templates(
                    working_df,
                    top_k=self.population_top_k,
                    result_cb=_emit_streamed_variant,
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
                    params = dict(row["params"])
                    context = {
                        "ctx_high_vol_avg_return": row.get("ctx_high_vol_avg_return", 0.0),
                        "ctx_low_vol_avg_return": row.get("ctx_low_vol_avg_return", 0.0),
                        "ctx_trending_avg_return": row.get("ctx_trending_avg_return", 0.0),
                        "ctx_ranging_avg_return": row.get("ctx_ranging_avg_return", 0.0),
                        "ctx_trend_confidence": row.get("ctx_trend_confidence", 0.0),
                        "ctx_volatility_confidence": row.get("ctx_volatility_confidence", 0.0),
                        "ctx_sample_count": row.get("ctx_sample_count", 0),
                        "ctx_return_scale": row.get("ctx_return_scale", 0.0),
                    }
                    explanation = self._strategy_explanation(str(row["template_key"]), params, context=context)
                    key_sig = (str(row["template_key"]), str(sorted(params.items())))
                    sig = f"{row['template_key']}|{json.dumps(params, sort_keys=True)}"
                    payload = {
                        "id": strategy_ids.get(sig, f"GEN{gen}-{strategy_counter:04d}"),
                        "generation": gen,
                        "name": str(row["strategy"]),
                        "family": str(row["template_key"]),
                        "origin": str(row.get("origin", "random")),
                        "fitness": round(float(row["fitness"]), 4),
                        "robustness": round(float(row["robustness_score"]), 4),
                        "validation": round(float(row["robustness_score"]), 4),
                        "status": "survived" if key_sig in survivors else "rejected",
                        "timeframe": "active",
                        "entry": explanation["entry"],
                        "exit": explanation["exit"],
                        "indicators": explanation["indicators"],
                        "explanation": explanation["summary"],
                        "regime": explanation["regime"],
                        "params": params,
                        "trade_count": int(row.get("test_trades", 0)),
                        "win_rate": round(float(row.get("test_win_rate_pct", 0.0)), 2),
                        "pnl": round(float(row.get("test_return_pct", 0.0)), 4),
                        "drawdown": round(float(row.get("test_max_drawdown_pct", 0.0)), 4),
                        "avg_trade_return": round(float(row.get("test_avg_trade_return_pct", 0.0)), 4),
                        "max_win": round(float(row.get("test_max_win_pct", 0.0)), 4),
                        "max_loss": round(float(row.get("test_max_loss_pct", 0.0)), 4),
                        "trade_distribution": f"W{int(row.get('test_win_trades', 0))}/L{int(row.get('test_loss_trades', 0))}",
                        "performance_context": str(row.get("performance_context", "")),
                        "ctx_high_vol_avg_return": round(float(row.get("ctx_high_vol_avg_return", 0.0)), 4),
                        "ctx_low_vol_avg_return": round(float(row.get("ctx_low_vol_avg_return", 0.0)), 4),
                        "ctx_trending_avg_return": round(float(row.get("ctx_trending_avg_return", 0.0)), 4),
                        "ctx_ranging_avg_return": round(float(row.get("ctx_ranging_avg_return", 0.0)), 4),
                        "context_confidence": round(float(row.get("ctx_confidence", 0.0)), 4),
                        "trend_context_confidence": round(float(row.get("ctx_trend_confidence", 0.0)), 4),
                        "volatility_context_confidence": round(float(row.get("ctx_volatility_confidence", 0.0)), 4),
                        "behavior_robustness": round(float(row.get("behavior_robustness", 0.0)), 2),
                        "time_stability": round(float(row.get("ctx_time_stability", 0.0)), 4),
                        "decay_score": round(float(row.get("ctx_decay_score", 0.0)), 4),
                        "decay_flag": bool(row.get("ctx_decay_flag", False)),
                        "ctx_return_scale": round(float(row.get("ctx_return_scale", 0.0)), 6),
                    }
                    self.strategy.emit(payload)

                seed_pool = top_variants[["template_key", "params"]].to_dict("records")
                self.log.emit("INFO", f"backtest completed: generation={gen} best_fitness={float(best['fitness']):.2f}")

            self.log.emit("INFO", f"strategy generation completed: total_candidates={strategy_counter}")

            if last_top is None:
                raise RuntimeError("Evolution completed with no best strategy")

            self.stage.emit("Validation")
            self.log.emit("INFO", "validation started")
            validation_targets = top_variants.head(min(5, len(top_variants)))
            wf = None
            stability = 0.0
            for v_idx, (_, vrow) in enumerate(validation_targets.iterrows(), start=1):
                if self._cancel:
                    return
                params = dict(vrow["params"])
                context = {
                    "ctx_high_vol_avg_return": vrow.get("ctx_high_vol_avg_return", 0.0),
                    "ctx_low_vol_avg_return": vrow.get("ctx_low_vol_avg_return", 0.0),
                    "ctx_trending_avg_return": vrow.get("ctx_trending_avg_return", 0.0),
                    "ctx_ranging_avg_return": vrow.get("ctx_ranging_avg_return", 0.0),
                    "ctx_trend_confidence": vrow.get("ctx_trend_confidence", 0.0),
                    "ctx_volatility_confidence": vrow.get("ctx_volatility_confidence", 0.0),
                    "ctx_sample_count": vrow.get("ctx_sample_count", 0),
                    "ctx_return_scale": vrow.get("ctx_return_scale", 0.0),
                }
                explanation = self._strategy_explanation(str(vrow["template_key"]), params, context=context)
                self.log.emit("INFO", f"validation running [{v_idx}/{len(validation_targets)}] {vrow['template_key']} params={params}")
                v_wf, v_stability = walk_forward_validate(
                    working_df,
                    template_key=str(vrow["template_key"]),
                    params=params,
                    folds=4,
                )
                sig = f"{vrow['template_key']}|{json.dumps(params, sort_keys=True)}"
                strategy_id = f"G{self.generations}-{hashlib.md5(sig.encode('utf-8')).hexdigest()[:10]}"
                self.strategy.emit({
                    "id": strategy_id,
                    "generation": self.generations,
                    "name": str(vrow["strategy"]),
                    "family": str(vrow["template_key"]),
                    "origin": str(vrow.get("origin", "random")),
                    "fitness": round(float(vrow["fitness"]), 4),
                    "robustness": round(float(vrow["robustness_score"]), 4),
                    "validation": round(float(v_stability), 4),
                    "status": "validated",
                    "timeframe": "active",
                    "entry": explanation["entry"],
                    "exit": explanation["exit"],
                    "indicators": explanation["indicators"],
                    "explanation": explanation["summary"],
                    "regime": explanation["regime"],
                    "params": params,
                    "trade_count": int(vrow.get("test_trades", 0)),
                    "win_rate": round(float(vrow.get("test_win_rate_pct", 0.0)), 2),
                    "pnl": round(float(vrow.get("test_return_pct", 0.0)), 4),
                    "drawdown": round(float(vrow.get("test_max_drawdown_pct", 0.0)), 4),
                    "avg_trade_return": round(float(vrow.get("test_avg_trade_return_pct", 0.0)), 4),
                    "max_win": round(float(vrow.get("test_max_win_pct", 0.0)), 4),
                    "max_loss": round(float(vrow.get("test_max_loss_pct", 0.0)), 4),
                    "trade_distribution": f"W{int(vrow.get('test_win_trades', 0))}/L{int(vrow.get('test_loss_trades', 0))}",
                    "performance_context": str(vrow.get("performance_context", "")),
                    "ctx_high_vol_avg_return": round(float(vrow.get("ctx_high_vol_avg_return", 0.0)), 4),
                    "ctx_low_vol_avg_return": round(float(vrow.get("ctx_low_vol_avg_return", 0.0)), 4),
                    "ctx_trending_avg_return": round(float(vrow.get("ctx_trending_avg_return", 0.0)), 4),
                    "ctx_ranging_avg_return": round(float(vrow.get("ctx_ranging_avg_return", 0.0)), 4),
                    "context_confidence": round(float(vrow.get("ctx_confidence", 0.0)), 4),
                    "trend_context_confidence": round(float(vrow.get("ctx_trend_confidence", 0.0)), 4),
                    "volatility_context_confidence": round(float(vrow.get("ctx_volatility_confidence", 0.0)), 4),
                    "behavior_robustness": round(float(vrow.get("behavior_robustness", 0.0)), 2),
                    "time_stability": round(float(vrow.get("ctx_time_stability", 0.0)), 4),
                    "decay_score": round(float(vrow.get("ctx_decay_score", 0.0)), 4),
                    "decay_flag": bool(vrow.get("ctx_decay_flag", False)),
                    "ctx_return_scale": round(float(vrow.get("ctx_return_scale", 0.0)), 6),
                })
                self.log.emit("INFO", f"validation completed [{v_idx}/{len(validation_targets)}] stability={float(v_stability):.2f}")
                if v_idx == 1:
                    wf = v_wf
                    stability = float(v_stability)

            if wf is None:
                wf = pd.DataFrame()
            self.log.emit("INFO", f"validation completed: walk_forward_stability={stability:.2f}")

            self.stage.emit("AI analysis")
            self.log.emit("INFO", "AI analysis started")

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
                self.log.emit(
                    "INFO",
                    f"ai epoch {int(epoch)}/{int(total)} loss={float(loss):.4f} acc={float(acc):.4f} "
                    f"val_loss={float(extra.get('val_loss', loss)):.4f} val_acc={float(extra.get('val_acc', acc)):.4f}"
                )

            ai = analyze_market_ai(working_df, model_type=self.model_type, epoch_cb=_epoch_cb)
            self.log.emit("INFO", "AI analysis completed")

            self.stage.emit("Finalizing results")
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
            self.log.emit("INFO", "final results ready")
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
    chartCandlesChanged = Signal()
    chartTimeframeChanged = Signal()
    chartWindowChanged = Signal()
    previewRowsChanged = Signal()
    previewColumnsChanged = Signal()
    featureMetaChanged = Signal()

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
        self._chart_timeframe = "1s"
        self._chart_candles: list[dict] = []
        self._chart_all_candles: list[dict] = []
        self._chart_window_size = 220
        self._chart_window_end = 0
        self._preview_rows: list[dict] = []
        self._preview_columns: list[str] = []
        self._feature_df: pd.DataFrame | None = None
        self._feature_columns: list[str] = []
        self._feature_preview_rows: list[dict] = []
        self._feature_row_count = 0
        self._generated_feature_count = 0

        self._thread: QThread | None = None
        self._worker: ResearchWorker | None = None
        self._rank_tick = 0
        self._rank_tracker: dict[str, dict] = {}
        self._elite_pool: dict[str, dict] = {}

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


    @Property(str, notify=chartTimeframeChanged)
    def chartTimeframe(self):
        return self._chart_timeframe

    @Property("QVariantList", notify=chartCandlesChanged)
    def chartCandles(self):
        return self._chart_candles

    def _normalize_dataset_path(self, raw: str) -> str:
        value = (raw or "").strip()
        if not value:
            return ""
        parsed = urlparse(value)
        if parsed.scheme == "file":
            path = unquote(parsed.path or "")
            # Windows file URLs are usually /D:/...
            if path.startswith("/") and len(path) > 2 and path[2] == ":":
                path = path[1:]
            return path
        return unquote(value)

    @Slot(str)
    def logUiEvent(self, message: str):
        self._append_log("UI", message)


    @Property(int, notify=chartWindowChanged)
    def chartWindowSize(self):
        return self._chart_window_size

    @Property(int, notify=chartWindowChanged)
    def chartWindowEnd(self):
        return self._chart_window_end

    @Property("QVariantList", notify=previewRowsChanged)
    def previewRows(self):
        return self._preview_rows

    @Property("QVariantList", notify=previewColumnsChanged)
    def previewColumns(self):
        return self._preview_columns

    @Property("QVariantList", notify=featureMetaChanged)
    def featureColumns(self):
        return self._feature_columns

    @Property("QVariantList", notify=featureMetaChanged)
    def featurePreviewRows(self):
        return self._feature_preview_rows

    @Property(int, notify=featureMetaChanged)
    def featureRowCount(self):
        return self._feature_row_count

    @Property(int, notify=featureMetaChanged)
    def generatedFeatureCount(self):
        return self._generated_feature_count

    @Slot(str)
    def setDatasetPath(self, dataset_path: str):
        normalized = self._normalize_dataset_path(dataset_path)
        self._dataset_path = normalized
        self.datasetPathChanged.emit()
        if normalized:
            self._append_log("UI", f"Dataset path set: {normalized}")


    @Slot()
    def loadDataset(self):
        try:
            if not self._dataset_path:
                raise ValueError("No dataset selected")
            self._append_log("INFO", f"Loading dataset from: {self._dataset_path}")
            df, profile = load_market_file_minimal(self._dataset_path)
            self._base_df = df
            self._profile = asdict(profile)
            self._preview_columns = list(df.columns)
            self._preview_rows = df.head(20).astype(str).to_dict("records")
            self.previewColumnsChanged.emit()
            self.previewRowsChanged.emit()
            self.profileChanged.emit()
            self._feature_df = None
            self._feature_columns = []
            self._feature_preview_rows = []
            self._feature_row_count = 0
            self._generated_feature_count = 0
            self.featureMetaChanged.emit()
            self._refresh_chart_data()
            self._set_stage("Dataset loaded")
            self._append_log("INFO", f"Dataset loaded successfully: rows={profile.rows:,} cols={len(profile.columns)}")
            self._append_log("INFO", f"Columns: {', '.join(profile.columns)}")
            self._append_log("INFO", f"Range: {profile.start} -> {profile.end} | synthetic={profile.synthetic_pct:.2f}%")
        except Exception as exc:
            self._append_log("ERROR", f"Dataset load failed: {type(exc).__name__}: {exc}")



    @Slot()
    def generateFeatures(self):
        try:
            if self._base_df is None:
                raise ValueError("Load a dataset before generating features")
            groups = [
                "EMA", "SMA", "RSI", "MACD", "ATR", "BOLLINGER",
                "VOLATILITY", "VOLUME_SPIKE", "BREAKOUT", "CANDLE_RATIOS",
                "VWAP", "MOMENTUM", "ORDER_FLOW", "ZSCORE",
            ]
            self._append_log("INFO", "Generating features from loaded dataset")
            feature_df, generated_cols = generate_features(self._base_df, groups)
            self._feature_df = feature_df
            self._feature_columns = list(feature_df.columns)
            self._feature_preview_rows = feature_df.head(20).astype(str).to_dict("records")
            self._feature_row_count = int(len(feature_df))
            self._generated_feature_count = int(len(generated_cols))
            self.featureMetaChanged.emit()
            self._append_log("INFO", f"Features generated: {len(generated_cols)} new columns, rows={len(feature_df):,}")
            self._append_log("INFO", f"Feature columns: {', '.join(generated_cols[:30])}{' ...' if len(generated_cols) > 30 else ''}")
        except Exception as exc:
            self._append_log("ERROR", f"Feature generation failed: {type(exc).__name__}: {exc}")

    @Slot()
    def clearDataset(self):
        self._append_log("UI", "Clear dataset clicked")
        self._base_df = None
        self._dataset_path = ""
        self.datasetPathChanged.emit()
        self._profile = {}
        self.profileChanged.emit()
        self._preview_rows = []
        self._preview_columns = []
        self._feature_df = None
        self._feature_columns = []
        self._feature_preview_rows = []
        self._feature_row_count = 0
        self._generated_feature_count = 0
        self.previewRowsChanged.emit()
        self.previewColumnsChanged.emit()
        self.featureMetaChanged.emit()
        self._refresh_chart_data()
        self._set_stage("Dataset cleared")
        self._append_log("INFO", "Dataset cleared from shared app state")

    @Slot(str)
    def setChartTimeframe(self, timeframe: str):
        tf = timeframe if timeframe in {"1s", "1m", "5m", "15m", "30m", "1h", "2h", "4h"} else "1s"
        self._chart_timeframe = tf
        self._append_log("UI", f"Chart timeframe changed: {tf}")
        self.chartTimeframeChanged.emit()
        self._refresh_chart_data()

    @Slot(int)
    def panChart(self, delta: int):
        if not self._chart_all_candles:
            return
        self._chart_window_end = max(self._chart_window_size, min(len(self._chart_all_candles), self._chart_window_end + int(delta)))
        self._update_chart_window()

    @Slot(int)
    def zoomChart(self, direction: int):
        if direction > 0:
            self._chart_window_size = max(40, int(self._chart_window_size * 0.8))
        else:
            self._chart_window_size = min(1200, int(self._chart_window_size * 1.25))
        if self._chart_all_candles:
            self._chart_window_end = max(self._chart_window_size, min(len(self._chart_all_candles), self._chart_window_end))
        self._update_chart_window()

    def _update_chart_window(self):
        if not self._chart_all_candles:
            self._chart_candles = []
            self.chartCandlesChanged.emit()
            self.chartWindowChanged.emit()
            return
        end = max(self._chart_window_size, min(len(self._chart_all_candles), self._chart_window_end or len(self._chart_all_candles)))
        start = max(0, end - self._chart_window_size)
        self._chart_window_end = end
        self._chart_candles = self._chart_all_candles[start:end]
        self.chartCandlesChanged.emit()
        self.chartWindowChanged.emit()

    def _refresh_chart_data(self):
        if self._base_df is None:
            self._chart_all_candles = []
            self._chart_candles = []
            self._chart_window_end = 0
            self.chartCandlesChanged.emit()
            self.chartWindowChanged.emit()
            return

        self._chart_all_candles = build_candle_payload(self._base_df, timeframe=self._chart_timeframe, window=None)
        self._chart_window_end = len(self._chart_all_candles)
        self._update_chart_window()

    @Slot()
    def startResearch(self):
        if self._thread is not None:
            self._append_log("WARN", "Research already running")
            return
        self._append_log("UI", "Start clicked")
        self._strategies = []
        self._selected_strategy = {}
        self._fitness_series = []
        self._loss_series = []
        self._accuracy_series = []
        self._val_loss_series = []
        self._val_accuracy_series = []
        self._regime_counts = {}
        self._feature_importance = {}
        self._rank_tick = 0
        self._rank_tracker = {}
        self._elite_pool = {}
        self.strategiesChanged.emit()
        self.selectedStrategyChanged.emit()
        self.fitnessSeriesChanged.emit()
        self.lossSeriesChanged.emit()
        self.accuracySeriesChanged.emit()
        self.valLossSeriesChanged.emit()
        self.valAccuracySeriesChanged.emit()

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
        self._append_log("UI", "Pause clicked")
        self._append_log("WARN", "Pause is not implemented for QML orchestrator yet")

    @Slot()
    def stopResearch(self):
        self._append_log("UI", "Stop clicked")
        if self._worker is not None:
            self._worker.cancel()
        self._append_log("INFO", "Cancellation requested")

    @Slot(str)
    def selectStrategyById(self, strategy_id: str):
        for row in self._strategies:
            if row.get("id") == strategy_id:
                self._selected_strategy = row
                self.selectedStrategyChanged.emit()
                self._append_log("UI", f"Strategy selected: {strategy_id}")
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

    def _compute_strategy_score(self, row: dict) -> float:
        pnl = float(row.get("pnl", 0.0) or 0.0)
        win_rate = float(row.get("win_rate", 0.0) or 0.0)
        drawdown = abs(float(row.get("drawdown", 0.0) or 0.0))
        trades = int(row.get("trade_count", 0) or 0)

        # Real-score blend: reward return/win-rate/stability, penalize drawdown.
        stability_bonus = min(1.0, trades / 40.0) * 10.0
        return round(pnl * 1.0 + win_rate * 0.35 - drawdown * 0.75 + stability_bonus, 4)

    def _resort_and_rank_strategies(self):
        if not self._strategies:
            return
        self._rank_tick += 1
        for row in self._strategies:
            row["score"] = self._compute_strategy_score(row)
        self._strategies.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
        for i, row in enumerate(self._strategies, start=1):
            row["rank"] = i

        n = len(self._strategies)
        population_maturity = float(1.0 - (1.0 / np.sqrt(n + 1.0)))
        scores = np.array([float(r.get("score", 0.0)) for r in self._strategies], dtype=float)
        score_std = float(np.std(scores, ddof=0))
        score_median = float(np.median(scores))
        top_score = float(scores[0])
        robust_vals = np.array([float(r.get("behavior_robustness", 0.0)) for r in self._strategies], dtype=float)
        robust_median = float(np.median(robust_vals))
        top_n = max(1, int(np.sqrt(n)))

        for row in self._strategies:
            rank = int(row.get("rank", n))
            pct_raw = 1.0 if n <= 1 else float((n - rank) / (n - 1))
            pct = population_maturity * pct_raw + (1.0 - population_maturity) * 0.5
            row["population_maturity"] = round(population_maturity, 4)
            row["percentile_rank_raw"] = round(pct_raw, 4)
            row["percentile_rank"] = round(pct, 4)
            row["score_from_median_raw"] = round(float(row["score"]) - score_median, 4)
            row["score_gap_to_top_raw"] = round(top_score - float(row["score"]), 4)
            row["score_from_median"] = round(population_maturity * row["score_from_median_raw"], 4)
            row["score_gap_to_top"] = round(population_maturity * row["score_gap_to_top_raw"], 4)

            sid = str(row.get("id", ""))
            track = self._rank_tracker.get(sid, {"updates": 0, "rank_sum": 0.0, "rank_sq_sum": 0.0, "top_hits": 0})
            track["updates"] += 1
            track["rank_sum"] += rank
            track["rank_sq_sum"] += rank * rank
            if rank <= top_n:
                track["top_hits"] += 1
            self._rank_tracker[sid] = track

            upd = float(track["updates"])
            mean_rank = track["rank_sum"] / upd
            var_rank = max(0.0, track["rank_sq_sum"] / upd - mean_rank * mean_rank)
            rank_std = float(np.sqrt(var_rank))
            rank_consistency = 1.0 / (1.0 + (rank_std / (mean_rank + 1e-9)))
            top_duration = float(track["top_hits"] / upd)
            raw_stability = float(np.sqrt(max(0.0, rank_consistency * top_duration)))
            row["rank_stability_raw"] = round(raw_stability, 4)
            row["rank_stability"] = round(population_maturity * raw_stability + (1.0 - population_maturity) * 0.5, 4)

        stability_vals = np.array([float(r.get("rank_stability", 0.0)) for r in self._strategies], dtype=float)
        stab_median = float(np.median(stability_vals))
        composite = []
        for row in self._strategies:
            score_pct = row.get("percentile_rank", 0.0)
            robust_pct = float((robust_vals <= float(row.get("behavior_robustness", 0.0))).mean())
            stab_pct = float((stability_vals <= float(row.get("rank_stability", 0.0))).mean())
            dom_index = float((max(1e-9, score_pct) * max(1e-9, robust_pct) * max(1e-9, stab_pct)) ** (1.0 / 3.0))
            row["dominance_index"] = round(dom_index, 4)
            composite.append(dom_index)

        comp = np.array(composite, dtype=float)
        dom_std = float(np.std(comp, ddof=0))
        dominant_id = None
        if len(self._strategies) >= 2:
            best = self._strategies[0]
            second = self._strategies[1]
            gap = float(best.get("dominance_index", 0.0) - second.get("dominance_index", 0.0))
            gap_strength = gap / (dom_std + 1e-9)
            dominance_conf = (gap_strength / (1.0 + abs(gap_strength))) * population_maturity
            population_baseline = float(np.mean(comp))
            if gap > dom_std and float(best.get("behavior_robustness", 0.0)) >= robust_median and float(best.get("rank_stability", 0.0)) >= stab_median and dominance_conf > population_baseline:
                dominant_id = str(best.get("id", ""))
        for row in self._strategies:
            row["dominant_candidate"] = str(row.get("id", "")) == dominant_id
            row["dominant_provisional"] = bool(str(row.get("id", "")) == str(self._strategies[0].get("id", "")) and dominant_id is None and n > 1 and population_maturity < float(np.mean(comp)))

        # Rolling elite pool (persistent across cycles within a run).
        elite_target = max(1, int(round(np.sqrt(n))))
        score_pct_vals = np.array([float(r.get("percentile_rank", 0.0)) for r in self._strategies], dtype=float)
        robust_pct_vals = np.array([float((robust_vals <= float(r.get("behavior_robustness", 0.0))).mean()) for r in self._strategies], dtype=float)
        stab_pct_vals = np.array([float((stability_vals <= float(r.get("rank_stability", 0.0))).mean()) for r in self._strategies], dtype=float)
        elite_signal = (np.clip(score_pct_vals, 1e-9, 1.0) * np.clip(robust_pct_vals, 1e-9, 1.0) * np.clip(stab_pct_vals, 1e-9, 1.0)) ** (1.0 / 3.0)
        pool_scores: list[tuple[float, str]] = []
        for i, row in enumerate(self._strategies):
            sid = str(row.get("id", ""))
            prior = self._elite_pool.get(sid, {})
            tenure = int(prior.get("tenure", 0))
            tenure_boost = 1.0 - np.exp(-float(tenure) / (np.sqrt(n) + 1.0))
            stability_gate = float(np.clip(row.get("rank_stability", 0.0), 0.0, 1.0))
            value = float(elite_signal[i]) * (0.5 + 0.5 * stability_gate) * (1.0 + tenure_boost)
            pool_scores.append((value, sid))
        pool_scores.sort(key=lambda x: x[0], reverse=True)
        next_elite_ids = {sid for _, sid in pool_scores[:elite_target]}
        new_pool: dict[str, dict] = {}
        for row in self._strategies:
            sid = str(row.get("id", ""))
            if sid in next_elite_ids:
                prev_tenure = int(self._elite_pool.get(sid, {}).get("tenure", 0))
                new_pool[sid] = {"tenure": prev_tenure + 1}
                row["elite_status"] = True
            else:
                row["elite_status"] = False
        self._elite_pool = new_pool

        # Adaptive survival filter (mark-only; no deletions).
        robust_pct_vals = np.array(
            [float((robust_vals <= float(r.get("behavior_robustness", 0.0))).mean()) for r in self._strategies],
            dtype=float,
        )
        robust_raw_vals = np.array(
            [float(np.clip(float(r.get("behavior_robustness", 0.0)) / 100.0, 0.0, 1.0)) for r in self._strategies],
            dtype=float,
        )
        conf_vals = np.array([float(r.get("context_confidence", 0.0)) for r in self._strategies], dtype=float)
        conf_pct_vals = np.array([float((conf_vals <= float(v)).mean()) for v in conf_vals], dtype=float)
        conf_raw_vals = np.array([float(np.clip(v, 0.0, 1.0)) for v in conf_vals], dtype=float)
        decay_penalties = np.array(
            [float(r.get("decay_score", 0.0)) if bool(r.get("decay_flag", False)) else 0.0 for r in self._strategies],
            dtype=float,
        )
        decay_effect_scale = 0.5 + 0.5 * population_maturity
        decay_good_vals = 1.0 - np.clip(decay_penalties * decay_effect_scale, 0.0, 1.0)
        robust_signal = 0.5 * robust_pct_vals + 0.5 * robust_raw_vals
        conf_signal = 0.5 * conf_pct_vals + 0.5 * conf_raw_vals
        health = 0.4 * robust_signal + 0.4 * conf_signal + 0.2 * decay_good_vals
        q_filtered = min(0.49, population_maturity * population_maturity * 0.5)
        q_weak = min(0.85, (population_maturity * 0.5) + q_filtered)
        n_total = len(health)
        filtered_n = int(np.floor(q_filtered * n_total))
        weak_n = int(np.floor(q_weak * n_total))
        order = np.argsort(health)  # low health first
        filtered_set = set(order[:filtered_n].tolist())
        weak_set = set(order[filtered_n:weak_n].tolist())
        for i, row in enumerate(self._strategies):
            h = float(health[i])
            if i in filtered_set:
                row["survival_status"] = "filtered"
            elif i in weak_set:
                row["survival_status"] = "weak"
            else:
                row["survival_status"] = "active"
            row["survival_health"] = round(h, 4)

        # Diversity signal on top cohort.
        top_rows = self._strategies[:top_n]
        overall_unique = len({str(r.get("family", "")) for r in self._strategies})
        top_unique = len({str(r.get("family", "")) for r in top_rows})
        overall_div = overall_unique / max(1, n)
        top_div = top_unique / max(1, top_n)
        low_diversity = top_div < overall_div
        top_family_counts: dict[str, int] = {}
        for r in top_rows:
            fam = str(r.get("family", ""))
            top_family_counts[fam] = top_family_counts.get(fam, 0) + 1
        for row in self._strategies:
            fam = str(row.get("family", ""))
            row["diversity_warning"] = bool(low_diversity and top_family_counts.get(fam, 0) > 1)

    @Slot(object)
    def _on_strategy(self, payload: object):
        row = dict(payload)
        row_id = str(row.get("id", ""))
        replaced = False
        for i, existing in enumerate(self._strategies):
            if str(existing.get("id", "")) == row_id and row_id:
                merged = dict(existing)
                merged.update(row)
                self._strategies[i] = merged
                replaced = True
                break
        if not replaced:
            self._strategies.insert(0, row)
            self._strategies = self._strategies[:800]
        self._resort_and_rank_strategies()
        self.strategiesChanged.emit()
        if self._selected_strategy and row_id and str(self._selected_strategy.get("id", "")) == row_id:
            for existing in self._strategies:
                if str(existing.get("id", "")) == row_id:
                    self._selected_strategy = dict(existing)
                    self.selectedStrategyChanged.emit()
                    break
        elif not self._selected_strategy:
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
        self._set_stage("Final results ready")

        self._profile = dict(data.get("profile", {}))
        self.profileChanged.emit()
        self.previewRowsChanged.emit()
        self.previewColumnsChanged.emit()

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
