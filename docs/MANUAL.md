# Crypto Strategy Lab — User Manual (Automated AI Research Pipeline)

## 1) Quick Start
1. Install deps:
   - `pip install -r requirements.txt`
2. Launch app:
   - `python main.py`
3. Open **Data Lab** and load your refined dataset (`.parquet`/`.csv`).
4. Wait for initial profile/logs to confirm rows/date range loaded.
5. Go to **AI Lab**.
6. Select timeframe (start with **1m** for fastest workflow).
7. Keep defaults first run:
   - Population = 24
   - Generations = 12
   - Auto mode = ON
8. Click **Start**.

---

## 2) If Start feels "stuck"
This build now updates timeline continuously and uses automatic downsampling for very large runs.

If runs are still heavy:
- Use **1m** or **5m** timeframe first.
- Keep **Population <= 24**.
- Keep **Generations <= 12**.
- Keep all feature groups enabled first run; then reduce if needed.

Also check the timeline table and log panel for active stage updates.

---

## 3) AI Lab Controls
- **Start**: runs full pipeline (profile -> features -> evolution -> validation -> AI -> final package)
- **Pause/Resume**: pauses or resumes processing
- **Stop**: graceful cancel request
- **AI-Only Quick Run**: only regime/confidence model analysis
- **Open Live Monitor**: opens a dedicated real-time run window showing candidate test progress, generation evolution, and live engine logs
- **Neural Network Window**: opens automatically on run and displays architecture + topology map + live epoch chips + generation/candidate context

Settings:
- **Timeframe**: dataset slice resolution used for run
- **Population**: top candidates kept per generation sweep
- **Generations**: number of evolution rounds
- **Feature toggles**: indicator/feature groups used in auto feature engineering
  - Includes expanded groups: VWAP, Momentum, Order Flow, Z-Score, Donchian, Stochastic, Keltner, ADX, CCI, Williams %R, OBV, CMF, Ichimoku, Supertrend, Fractal, Microstructure

---

## 4) What happens in a full Start run
1. **Data understanding**
   - row count, range, synthetic ratio
2. **Feature engineering**
   - selected groups generated automatically
3. **Strategy evolution**
   - template variants tested, scored, and ranked (base grid + seeded mutations from prior winners)
   - best candidate per generation retained
4. **Validation**
   - walk-forward stability score computed
5. **AI analysis**
   - regime distribution
   - confidence and prediction distributions
   - loss/accuracy curves
6. **Final output**
   - best strategy card
   - TradingView replication text package

---

## 5) Reading results
### Pipeline Timeline
Shows active stage + percent + per-task notes.

### Generation Evolution
For each generation:
- best strategy
- fitness
- robustness
- stability
- return %
- drawdown %

### Best Strategy Card
Shows final promoted strategy details and key metrics.

### TradingView Replication Package
Contains deterministic rule summary and parameters for implementation in Pine Script.

### AI Panels
- Regime distribution
- Confidence distribution
- Prediction distribution
- Loss/accuracy curves
- Fitness-by-generation curve
- Live monitor also shows candidate throughput and an activity pulse chart

---

## 6) Practical first workflow (recommended)
1. Load dataset in Data Lab
2. Build timeframe **1m**
3. Run AI Lab Start with defaults
4. Inspect top strategy + generation scores
5. Move to Strategy Lab for deeper comparison/evolution sweep
6. Use Validation Lab for direct template walk-forward checks
7. Export/copy TradingView text from AI/Strategy outputs

---

## 7) Known current scope
This version is focused on research automation and visual pipeline activity.
It does **not** perform live execution.

---

## 8) Troubleshooting checklist
- No timeframe data in AI Lab:
  - switch timeframe, wait for cache build, re-open AI Lab
- Pipeline error popup:
  - check log panel line right before error
- Too slow:
  - lower generations/population and use higher timeframe
- Empty high-confidence setups:
  - this can happen on low-signal slices; try different timeframe window

---

## 9) Suggested next tuning for your machine
- Fast machine: Population 48+, Generations 20+
- Mid machine: Population 24, Generations 12
- Conservative machine: Population 12, Generations 6
