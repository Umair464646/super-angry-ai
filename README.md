# Crypto Strategy Lab

Crypto Strategy Lab runs with a **PySide6 + QML** front end and Python backend engines.

## Current QML execution flow (real, non-demo)
The QML shell now drives actual backend processing via `AppState` + `ResearchWorker`:
1. Load real CSV/Parquet dataset from the dataset path field.
2. Build real features using `generate_features`.
3. Run real strategy evolution via `evolve_templates`.
4. Run walk-forward validation via `walk_forward_validate`.
5. Run AI analysis via `analyze_market_ai`.
6. Stream real strategy rows, logs, stage text, and AI curves to QML.

## UI Architecture
- Left navigation rail
- Top command bar (dataset path + Start/Pause/Stop + stage/model status)
- Main tab workspace (Overview, Strategies, Evolution, Neural, Results)
- Right inspector panel (selected strategy details + copy)
- Bottom collapsible log console

## Run
```bash
pip install -r requirements.txt
python main.py
```

## Important
- No random strategy/timer simulation is used in the active QML flow.
- If no valid dataset path is set, start will fail with an explicit log error.
