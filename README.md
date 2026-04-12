# Crypto Strategy Lab

Crypto Strategy Lab is now launched through a **PySide6 + QML (Qt Quick Controls 2)** shell focused on a premium desktop workspace layout.

## UI Architecture (QML-first)
- Left navigation rail (Home, Data, Strategy, Evolution, Neural, Backtest, Results, Export)
- Top command bar (project, dataset, timeframe, Start/Pause/Stop, model state)
- Main tab workspace (Overview, Strategies, Evolution, Neural, Results)
- Right inspector panel (selected strategy details + copy action)
- Bottom collapsible log console

## Included Reusable QML Components
- `NavigationRail`
- `TopBar`
- `StrategyItem`
- `ChartPanel`
- `InspectorPanel`
- `LogConsole`

## Run
```bash
pip install -r requirements.txt
python main.py
```

## Notes
- The QML UI is designed to be responsive, scrollable, and visually clean.
- Real-time demo activity is powered by `AppState` (`app/ui/qml_app.py`) so charts, strategy feed, and logs update live.
