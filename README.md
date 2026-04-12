# Crypto Strategy Lab V9 Feature Lab

Desktop build with:
- optimized large-dataset loading
- disk-cached timeframes
- chart workspace
- Feature Lab with real feature generation
- exportable feature datasets
- Strategy Lab candidate generation and scoring
- Backtest Lab with realistic fills/fees/slippage
- Validation Lab walk-forward stability scoring
- AI Lab with regime classification, setup confidence scoring, and training curves
- AI Lab automated Start pipeline (profile -> features -> evolution -> validation -> AI -> TradingView package)
- Extended AI Lab sweeps: up to 200 generations, larger populations, and wider per-generation variant exploration
- AI Lab Live Monitor window for real-time candidate testing and evolution visibility
- Neural Network Training window with topology visualization, live epoch chips, and pipeline generation/candidate context
- Expanded indicator set (VWAP, Momentum, Order Flow, Z-Score, Donchian, Stochastic, Keltner, ADX, CCI, Williams %R, OBV, CMF, Ichimoku, Supertrend, Fractal, Microstructure)

## Run
pip install -r requirements.txt
python main.py

## Manual
See `docs/MANUAL.md` for a step-by-step operating guide.
