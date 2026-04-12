from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from app.core.feature_engine import generate_features
from app.core.strategy_engine import evolve_templates, walk_forward_validate, tradingview_strategy_text
from app.core.ai_engine import analyze_market_ai


@dataclass
class ResearchRunConfig:
    selected_features: list[str]
    generations: int = 4
    population_top_k: int = 12
    max_variants_per_generation: int = 600
    validation_folds: int = 4
    max_rows_for_evolution: int = 6_000
    max_rows_for_ai: int = 80_000
    model_type: str = "mlp"


class AutoResearchWorker(QObject):
    progress = pyqtSignal(int)
    stage = pyqtSignal(str)
    log = pyqtSignal(str, str)
    timeline = pyqtSignal(str, int, str)  # stage_name, percent, note
    generation = pyqtSignal(int, int, float, int)  # gen, survivors, best_fitness, population
    candidate_test = pyqtSignal(int, int, int, str)  # gen, done, total, family
    ai_epoch = pyqtSignal(int, int, float, float, object)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, df: pd.DataFrame, config: ResearchRunConfig):
        super().__init__()
        self.df = df
        self.config = config
        self._cancel = False
        self._pause = False

    def cancel(self):
        self._cancel = True

    def set_paused(self, paused: bool):
        self._pause = bool(paused)

    def _checkpoint(self):
        while self._pause and not self._cancel:
            time.sleep(0.1)
        if self._cancel:
            raise RuntimeError("Research run cancelled")

    @pyqtSlot()
    def run(self):
        try:
            if self.df is None or len(self.df) == 0:
                raise ValueError("No dataframe available for automated research")

            self.stage.emit("Step A: Data understanding")
            self.timeline.emit("Data understanding", 100, "profile computed")
            self.progress.emit(5)
            self._checkpoint()

            profile = {
                "rows": int(len(self.df)),
                "columns": int(len(self.df.columns)),
                "start": str(self.df["timestamp"].iloc[0]) if "timestamp" in self.df.columns else "n/a",
                "end": str(self.df["timestamp"].iloc[-1]) if "timestamp" in self.df.columns else "n/a",
                "synthetic_ratio": float(self.df["synthetic"].fillna(0).mean()) if "synthetic" in self.df.columns else 0.0,
            }
            self.log.emit("INFO", f"Dataset rows={profile['rows']:,} cols={profile['columns']}")
            self.log.emit("INFO", f"Synthetic ratio={profile['synthetic_ratio']:.2%}")

            self.stage.emit("Step B/C: Feature engineering and QC")
            self.timeline.emit("Feature engineering", 20, "building selected feature groups")
            self.progress.emit(15)
            self._checkpoint()

            featured_df, generated_cols = generate_features(self.df, self.config.selected_features)
            self.timeline.emit("Feature engineering", 100, f"generated {len(generated_cols)} columns")
            self.progress.emit(35)
            self._checkpoint()

            self.stage.emit("Step D/E/F/G: Strategy generation, backtest, validation, evolution")
            self.timeline.emit("Strategy evolution", 10, "initializing population")

            all_generations = []
            best_rows = []
            seed_pool: list[dict] = []
            current_df = featured_df
            if "synthetic" in current_df.columns:
                synthetic_ratio = float(current_df["synthetic"].fillna(0).mean())
                if synthetic_ratio > 0.40:
                    filtered = current_df[current_df["synthetic"].fillna(0).astype(int) == 0]
                    if len(filtered) >= 500:
                        self.log.emit(
                            "WARN",
                            f"Synthetic-heavy slice detected ({synthetic_ratio:.2%}); evolution uses non-synthetic rows: {len(current_df):,} -> {len(filtered):,}",
                        )
                        current_df = filtered.reset_index(drop=True)
            if len(current_df) > self.config.max_rows_for_evolution:
                stride = max(1, len(current_df) // self.config.max_rows_for_evolution)
                current_df = current_df.iloc[::stride].reset_index(drop=True)
                self.log.emit(
                    "WARN",
                    f"Evolution input downsampled for speed: {len(featured_df):,} -> {len(current_df):,} rows (stride {stride})",
                )

            for gen in range(1, self.config.generations + 1):
                self._checkpoint()

                def _variant_progress(done: int, total: int, family: str):
                    pct = int((done / max(1, total)) * 100)
                    self.timeline.emit("Strategy evolution", pct, f"gen {gen}: {family} {done}/{total}")
                    self.candidate_test.emit(gen, done, total, family)
                    self._checkpoint()

                all_variants, top_variants = evolve_templates(
                    current_df,
                    top_k=self.config.population_top_k,
                    progress_cb=_variant_progress,
                    seed_pool=seed_pool,
                    max_variants=self.config.max_variants_per_generation,
                )
                best = top_variants.iloc[0]
                seed_pool = top_variants[["template_key", "params"]].to_dict("records")

                wf, stability = walk_forward_validate(
                    current_df,
                    template_key=str(best["template_key"]),
                    params=dict(best["params"]),
                    folds=self.config.validation_folds,
                )

                fitness = float(best["fitness"])
                survivors = int(len(top_variants))
                population = int(len(all_variants))

                record = {
                    "generation": gen,
                    "best_strategy": str(best["strategy"]),
                    "template_key": str(best["template_key"]),
                    "params": dict(best["params"]),
                    "fitness": fitness,
                    "robustness_score": float(best["robustness_score"]),
                    "test_return_pct": float(best["test_return_pct"]),
                    "test_win_rate_pct": float(best["test_win_rate_pct"]),
                    "test_max_drawdown_pct": float(best["test_max_drawdown_pct"]),
                    "stability_score": float(stability),
                    "population": population,
                    "survivors": survivors,
                }
                best_rows.append(record)
                all_generations.append({"generation": gen, "all": all_variants, "top": top_variants, "wf": wf})

                self.generation.emit(gen, survivors, fitness, population)
                pct = 35 + int((gen / max(1, self.config.generations)) * 45)
                self.progress.emit(min(80, pct))
                self.timeline.emit("Strategy evolution", int((gen / self.config.generations) * 100), f"generation {gen} complete")
                self.log.emit("INFO", f"Generation {gen}: best fitness={fitness:.2f} stability={stability:.2f}")

            self._checkpoint()
            self.stage.emit("Step H/I: AI regime and confidence analysis")
            self.timeline.emit("AI analysis", 15, "training regime/confidence model")
            self.progress.emit(85)

            ai_df = featured_df
            if len(ai_df) > self.config.max_rows_for_ai:
                stride = max(1, len(ai_df) // self.config.max_rows_for_ai)
                ai_df = ai_df.iloc[::stride].reset_index(drop=True)
                self.log.emit(
                    "WARN",
                    f"AI input downsampled for speed: {len(featured_df):,} -> {len(ai_df):,} rows (stride {stride})",
                )
            ai_result = analyze_market_ai(
                ai_df,
                model_type=self.config.model_type,
                epoch_cb=lambda e, total, loss, acc, extra=None: self.ai_epoch.emit(e, total, loss, acc, extra or {}),
            )
            self.timeline.emit("AI analysis", 100, "AI outputs ready")

            self.stage.emit("Step J: Final ranking and export package")
            self.progress.emit(98)

            best_df = pd.DataFrame(best_rows).sort_values(
                ["fitness", "stability_score", "robustness_score"], ascending=False
            ).reset_index(drop=True)
            best_strategy = best_df.iloc[0].to_dict()

            tv_text = tradingview_strategy_text(
                template_key=str(best_strategy["template_key"]),
                params=dict(best_strategy["params"]),
            )

            payload = {
                "profile": profile,
                "generated_features": generated_cols,
                "best_by_generation": best_df,
                "ai": ai_result,
                "tradingview_text": tv_text,
                "top_strategy": best_strategy,
            }

            self.progress.emit(100)
            self.stage.emit("Automated research complete")
            self.finished.emit(payload)

        except Exception as exc:
            self.progress.emit(0)
            self.stage.emit("Automated research failed")
            self.error.emit(str(exc))
            self.log.emit("ERROR", str(exc))
