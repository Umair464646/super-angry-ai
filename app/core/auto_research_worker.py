from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from app.core.feature_engine import generate_features
from app.core.strategy_engine import evolve_templates, walk_forward_validate, tradingview_strategy_text, TEMPLATES
from app.core.ai_engine import analyze_market_ai


def _infer_strategy_profile(template_key: str, params: dict) -> dict:
    key = (template_key or "").lower()
    family = "Composite"
    regime = "mixed"
    modules = ["Regime", "Structure", "Flow", "Timing", "Risk", "Filters"]
    risk_model = "ATR stop + fixed RR"
    notes = "Deterministic signals only; avoid synthetic-triggered entries."

    if "ema" in key or "trend" in key:
        family = "Trend Continuation"
        regime = "trend / volatility expansion"
        risk_model = "ATR stop + trend trailing"
    elif "breakout" in key:
        family = "Breakout Expansion"
        regime = "compression release / breakout phase"
        risk_model = "structure stop + adaptive target"
    elif "reversal" in key or "rsi" in key:
        family = "Mean Reversion"
        regime = "range / failed breakout"
        risk_model = "structure stop + time stop"
    elif "vwap" in key:
        family = "Order Flow Reclaim"
        regime = "intraday trend with participation"
        risk_model = "VWAP invalidation + ATR emergency exit"

    return {
        "family": family,
        "regime_suitability": regime,
        "modules_used": modules,
        "risk_model": risk_model,
        "notes": notes,
        "tradingview_replication_notes": "Use bar-close confirmation with next-bar execution assumptions.",
        "parameters": params,
    }


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
    strategy_event = pyqtSignal(object)
    mutation_event = pyqtSignal(object)
    lifecycle_event = pyqtSignal(object)
    evolution_diag = pyqtSignal(object)
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
            best_fitness_so_far = -1e18
            stagnation_count = 0
            template_map = {t.key: t for t in TEMPLATES}
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

                exploration_strength = 0.0
                if stagnation_count >= 2:
                    exploration_strength = min(0.8, 0.25 + 0.15 * (stagnation_count - 1))
                    self.log.emit("WARN", f"[AI][GEN {gen}] Stagnation detected ({stagnation_count}), forcing exploration strength={exploration_strength:.2f}")
                all_variants, top_variants = evolve_templates(
                    current_df,
                    top_k=self.config.population_top_k,
                    progress_cb=_variant_progress,
                    seed_pool=seed_pool,
                    max_variants=self.config.max_variants_per_generation,
                    exploration_strength=exploration_strength,
                )
                best = top_variants.iloc[0]
                survivors_keys = set(
                    (str(r["template_key"]), str(sorted(dict(r["params"]).items())))
                    for _, r in top_variants.iterrows()
                )
                elite_count = min(3, len(top_variants))
                lifecycle_counts = {
                    "generated": int(len(all_variants)),
                    "backtested": int(len(all_variants)),
                    "validated": int(len(all_variants)),
                    "survived": int(len(top_variants)),
                    "rejected": int(len(all_variants) - len(top_variants)),
                    "mutated": int(len(top_variants)),
                    "archived": int(elite_count),
                }
                self.lifecycle_event.emit(lifecycle_counts)
                for idx, (_, row) in enumerate(all_variants.iterrows(), start=1):
                    tkey = str(row["template_key"])
                    tmpl = template_map.get(tkey)
                    skey = (tkey, str(sorted(dict(row["params"]).items())))
                    status = "survived" if skey in survivors_keys else "rejected"
                    profile_info = _infer_strategy_profile(tkey, dict(row["params"]))
                    ev = {
                        "strategy_id": f"GEN{gen}-{idx:04d}",
                        "generation": gen,
                        "name": str(row["strategy"]),
                        "type": tkey,
                        "family": profile_info["family"],
                        "regime_suitability": profile_info["regime_suitability"],
                        "timeframe": "active",
                        "indicators": ", ".join(tmpl.indicators) if tmpl else tkey,
                        "modules_used": profile_info["modules_used"],
                        "parameters": dict(row["params"]),
                        "entry_logic": tmpl.entry_logic if tmpl else "n/a",
                        "exit_logic": tmpl.exit_logic if tmpl else "n/a",
                        "filters": tmpl.filters if tmpl else "n/a",
                        "risk_model": profile_info["risk_model"],
                        "fitness": float(row["fitness"]),
                        "robustness": float(row["robustness_score"]),
                        "validation_score": float(row["robustness_score"]),
                        "status": status,
                        "tradingview_ready": "Yes",
                        "origin": str(row.get("origin", "random")),
                        "mutation_type": str(row.get("mutation_type", "base")),
                        "parent_strategy_id": str(row.get("parent_id", "none")),
                        "metrics": {
                            "return_pct": float(row["test_return_pct"]),
                            "drawdown_pct": float(row["test_max_drawdown_pct"]),
                            "trades": int(row["test_trades"]),
                        },
                        "notes": profile_info["notes"],
                        "tradingview_replication_notes": profile_info["tradingview_replication_notes"],
                    }
                    self.strategy_event.emit(ev)
                    self.log.emit(
                        "INFO",
                        f"[AI][GEN {gen}][CAND {idx}] Strategy created: {ev['name']} | fitness={ev['fitness']:.2f} robust={ev['robustness']:.2f} status={status}",
                    )

                if seed_pool and len(top_variants) > 0:
                    child = top_variants.iloc[0]
                    parent = seed_pool[0]
                    parent_params = dict(parent.get("params", {}))
                    child_params = dict(child["params"])
                    diffs = []
                    for k in sorted(set(parent_params) | set(child_params)):
                        if parent_params.get(k) != child_params.get(k):
                            diffs.append(f"{k}: {parent_params.get(k)} -> {child_params.get(k)}")
                    self.mutation_event.emit(
                        {
                            "parent_id": f"GEN{gen-1}-TOP",
                            "child_id": "GEN{gen}-TOP".format(gen=gen),
                            "mutation_type": "seeded_mutation",
                            "changes": diffs[:8],
                            "fitness_delta": float(child["fitness"]) - float(parent.get("fitness", 0.0)),
                            "robustness_delta": float(child["robustness_score"]) - float(parent.get("robustness_score", 0.0)),
                        }
                    )
                seed_pool = top_variants[["template_key", "params"]].to_dict("records")
                seed_pool[0]["fitness"] = float(best["fitness"])
                seed_pool[0]["robustness_score"] = float(best["robustness_score"])

                wf, stability = walk_forward_validate(
                    current_df,
                    template_key=str(best["template_key"]),
                    params=dict(best["params"]),
                    folds=self.config.validation_folds,
                )

                fitness = float(best["fitness"])
                if fitness > best_fitness_so_far + 0.05:
                    best_fitness_so_far = fitness
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                survivors = int(len(top_variants))
                population = int(len(all_variants))
                logic_div = float(all_variants["template_key"].nunique()) / max(1, population)
                param_div = float(all_variants["structure_sig"].nunique()) / max(1, population)
                indicator_div = logic_div
                diversity_score = round((logic_div * 0.45 + param_div * 0.45 + indicator_div * 0.10) * 100.0, 2)
                mutation_dist = all_variants["mutation_type"].value_counts().to_dict() if "mutation_type" in all_variants.columns else {}
                crossover_usage = int((all_variants.get("origin", pd.Series(dtype=str)) == "crossover").sum()) if "origin" in all_variants.columns else 0
                self.evolution_diag.emit(
                    {
                        "generation": gen,
                        "diversity_score": diversity_score,
                        "logic_diversity": round(logic_div * 100.0, 2),
                        "parameter_diversity": round(param_div * 100.0, 2),
                        "mutation_distribution": mutation_dist,
                        "crossover_usage": crossover_usage,
                        "stagnation_count": stagnation_count,
                        "exploration_strength": exploration_strength,
                        "exploration_vs_exploitation": round(exploration_strength / max(0.01, 1 - exploration_strength), 2) if exploration_strength > 0 else 0.0,
                    }
                )

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
