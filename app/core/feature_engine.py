from __future__ import annotations
import numpy as np
import pandas as pd

DEFAULT_EMA_PERIODS = [9, 20, 50, 100, 200]
DEFAULT_SMA_PERIODS = [20, 50, 100, 200]

def ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("timestamp").reset_index(drop=True)

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()

def compute_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return tr, atr

def compute_bollinger(close: pd.Series, period: int = 20, std_mult: float = 2.0):
    basis = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper = basis + std_mult * std
    lower = basis - std_mult * std
    return basis, upper, lower, std

def add_ema_features(df: pd.DataFrame, periods=None) -> list[str]:
    periods = periods or DEFAULT_EMA_PERIODS
    cols = []
    for p in periods:
        name = f"ema_{p}"
        df[name] = compute_ema(df["close"], p)
        cols.append(name)
    return cols

def add_sma_features(df: pd.DataFrame, periods=None) -> list[str]:
    periods = periods or DEFAULT_SMA_PERIODS
    cols = []
    for p in periods:
        name = f"sma_{p}"
        df[name] = compute_sma(df["close"], p)
        cols.append(name)
    return cols

def add_rsi_features(df: pd.DataFrame) -> list[str]:
    df["rsi_14"] = compute_rsi(df["close"], 14)
    return ["rsi_14"]

def add_macd_features(df: pd.DataFrame) -> list[str]:
    macd, signal, hist = compute_macd(df["close"], 12, 26, 9)
    df["macd_12_26"] = macd
    df["macd_signal_9"] = signal
    df["macd_hist_12_26_9"] = hist
    return ["macd_12_26", "macd_signal_9", "macd_hist_12_26_9"]

def add_atr_features(df: pd.DataFrame) -> list[str]:
    tr, atr = compute_atr(df["high"], df["low"], df["close"], 14)
    df["true_range"] = tr
    df["atr_14"] = atr
    return ["true_range", "atr_14"]

def add_bollinger_features(df: pd.DataFrame) -> list[str]:
    basis, upper, lower, std = compute_bollinger(df["close"], 20, 2.0)
    df["bb_basis_20"] = basis
    df["bb_upper_20_2"] = upper
    df["bb_lower_20_2"] = lower
    df["bb_std_20"] = std
    df["bb_width_20_2"] = (upper - lower) / basis.replace(0, np.nan)
    return ["bb_basis_20", "bb_upper_20_2", "bb_lower_20_2", "bb_std_20", "bb_width_20_2"]

def add_volatility_features(df: pd.DataFrame) -> list[str]:
    returns = df["close"].pct_change()
    df["return_close"] = returns
    df["log_return_close"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility_20"] = returns.rolling(20, min_periods=20).std(ddof=0)
    df["volatility_50"] = returns.rolling(50, min_periods=50).std(ddof=0)
    return ["return_close", "log_return_close", "volatility_20", "volatility_50"]

def add_volume_spike_features(df: pd.DataFrame) -> list[str]:
    vol_ma_20 = df["volume"].rolling(20, min_periods=20).mean()
    vol_ma_50 = df["volume"].rolling(50, min_periods=50).mean()
    df["volume_ma_20"] = vol_ma_20
    df["volume_ma_50"] = vol_ma_50
    df["volume_spike_ratio_20"] = df["volume"] / vol_ma_20.replace(0, np.nan)
    df["volume_spike_ratio_50"] = df["volume"] / vol_ma_50.replace(0, np.nan)
    return ["volume_ma_20", "volume_ma_50", "volume_spike_ratio_20", "volume_spike_ratio_50"]

def add_breakout_features(df: pd.DataFrame) -> list[str]:
    prev_high_20 = df["high"].rolling(20, min_periods=20).max().shift(1)
    prev_low_20 = df["low"].rolling(20, min_periods=20).min().shift(1)
    df["rolling_high_20_prev"] = prev_high_20
    df["rolling_low_20_prev"] = prev_low_20
    breakout_up = (df["close"] > prev_high_20).astype("float")
    breakout_down = (df["close"] < prev_low_20).astype("float")
    if "synthetic" in df.columns:
        mask = df["synthetic"].fillna(0) == 1
        breakout_up = breakout_up.mask(mask, 0.0)
        breakout_down = breakout_down.mask(mask, 0.0)
    df["breakout_up_20"] = breakout_up
    df["breakout_down_20"] = breakout_down
    return ["rolling_high_20_prev", "rolling_low_20_prev", "breakout_up_20", "breakout_down_20"]

def add_candle_ratio_features(df: pd.DataFrame) -> list[str]:
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    body = (df["close"] - df["open"]).abs()
    upper = df["high"] - df[["open", "close"]].max(axis=1)
    lower = df[["open", "close"]].min(axis=1) - df["low"]
    df["candle_range"] = df.get("candle_range", df["high"] - df["low"])
    df["body_size"] = body
    df["upper_wick"] = df.get("upper_wick", upper)
    df["lower_wick"] = df.get("lower_wick", lower)
    df["body_to_range"] = body / rng
    df["upper_wick_to_range"] = upper / rng
    df["lower_wick_to_range"] = lower / rng
    return ["candle_range", "body_size", "upper_wick", "lower_wick", "body_to_range", "upper_wick_to_range", "lower_wick_to_range"]

FEATURE_BUILDERS = {
    "EMA": add_ema_features,
    "SMA": add_sma_features,
    "RSI": add_rsi_features,
    "MACD": add_macd_features,
    "ATR": add_atr_features,
    "BOLLINGER": add_bollinger_features,
    "VOLATILITY": add_volatility_features,
    "VOLUME_SPIKE": add_volume_spike_features,
    "BREAKOUT": add_breakout_features,
    "CANDLE_RATIOS": add_candle_ratio_features,
}


def add_vwap_features(df: pd.DataFrame) -> list[str]:
    if "vwap" in df.columns and df["vwap"].notna().any():
        vwap = df["vwap"]
    elif "quote_volume" in df.columns:
        vwap = (df["quote_volume"].fillna(0).cumsum() / df["volume"].replace(0, np.nan).cumsum()).ffill()
    else:
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        vwap = ((tp * df["volume"].fillna(0)).cumsum() / df["volume"].replace(0, np.nan).cumsum()).ffill()
    df["vwap_auto"] = vwap
    df["close_to_vwap"] = (df["close"] - vwap) / vwap.replace(0, np.nan)
    return ["vwap_auto", "close_to_vwap"]


def add_momentum_features(df: pd.DataFrame) -> list[str]:
    df["mom_3"] = df["close"].pct_change(3)
    df["mom_10"] = df["close"].pct_change(10)
    df["roc_20"] = ((df["close"] / df["close"].shift(20)) - 1.0) * 100.0
    return ["mom_3", "mom_10", "roc_20"]


def add_orderflow_features(df: pd.DataFrame) -> list[str]:
    cols = []
    if "buy_volume" in df.columns and "sell_volume" in df.columns:
        total = (df["buy_volume"] + df["sell_volume"]).replace(0, np.nan)
        df["buy_sell_imbalance_of"] = (df["buy_volume"] - df["sell_volume"]) / total
        cols.append("buy_sell_imbalance_of")
    if "buy_sell_vol_delta" in df.columns:
        df["buy_sell_vol_delta_z"] = (
            (df["buy_sell_vol_delta"] - df["buy_sell_vol_delta"].rolling(100).mean()) /
            df["buy_sell_vol_delta"].rolling(100).std().replace(0, np.nan)
        )
        cols.append("buy_sell_vol_delta_z")
    return cols


def add_zscore_features(df: pd.DataFrame) -> list[str]:
    ma = df["close"].rolling(50).mean()
    sd = df["close"].rolling(50).std().replace(0, np.nan)
    df["close_z_50"] = (df["close"] - ma) / sd
    vma = df["volume"].rolling(50).mean()
    vsd = df["volume"].rolling(50).std().replace(0, np.nan)
    df["volume_z_50"] = (df["volume"] - vma) / vsd
    return ["close_z_50", "volume_z_50"]


def add_donchian_features(df: pd.DataFrame) -> list[str]:
    df["donchian_high_20"] = df["high"].rolling(20).max()
    df["donchian_low_20"] = df["low"].rolling(20).min()
    df["donchian_mid_20"] = (df["donchian_high_20"] + df["donchian_low_20"]) / 2.0
    return ["donchian_high_20", "donchian_low_20", "donchian_mid_20"]


def add_stochastic_features(df: pd.DataFrame) -> list[str]:
    low14 = df["low"].rolling(14).min()
    high14 = df["high"].rolling(14).max()
    k = ((df["close"] - low14) / (high14 - low14).replace(0, np.nan)) * 100.0
    d = k.rolling(3).mean()
    df["stoch_k_14"] = k
    df["stoch_d_3"] = d
    return ["stoch_k_14", "stoch_d_3"]


def add_keltner_features(df: pd.DataFrame) -> list[str]:
    ema20 = df["close"].ewm(span=20, adjust=False).mean()
    tr, atr = compute_atr(df["high"], df["low"], df["close"], 20)
    df["keltner_mid_20"] = ema20
    df["keltner_upper_20"] = ema20 + 2.0 * atr
    df["keltner_lower_20"] = ema20 - 2.0 * atr
    return ["keltner_mid_20", "keltner_upper_20", "keltner_lower_20"]


def add_adx_features(df: pd.DataFrame) -> list[str]:
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    tr, atr = compute_atr(high, low, close, 14)
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr.replace(0, np.nan))
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    df["plus_di_14"] = plus_di
    df["minus_di_14"] = minus_di
    df["adx_14"] = adx
    return ["plus_di_14", "minus_di_14", "adx_14"]


def add_cci_features(df: pd.DataFrame) -> list[str]:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    ma = tp.rolling(20).mean()
    md = (tp - ma).abs().rolling(20).mean().replace(0, np.nan)
    df["cci_20"] = (tp - ma) / (0.015 * md)
    return ["cci_20"]


def add_williams_r_features(df: pd.DataFrame) -> list[str]:
    hh = df["high"].rolling(14).max()
    ll = df["low"].rolling(14).min()
    df["williams_r_14"] = -100 * ((hh - df["close"]) / (hh - ll).replace(0, np.nan))
    return ["williams_r_14"]


def add_obv_features(df: pd.DataFrame) -> list[str]:
    direction = np.sign(df["close"].diff().fillna(0))
    df["obv"] = (direction * df["volume"]).cumsum()
    df["obv_ema_20"] = df["obv"].ewm(span=20, adjust=False).mean()
    return ["obv", "obv_ema_20"]


def add_cmf_features(df: pd.DataFrame) -> list[str]:
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"]).replace(0, np.nan)
    mfv = mfm * df["volume"]
    df["cmf_20"] = mfv.rolling(20).sum() / df["volume"].rolling(20).sum().replace(0, np.nan)
    return ["cmf_20"]


def add_ichimoku_features(df: pd.DataFrame) -> list[str]:
    conv = (df["high"].rolling(9).max() + df["low"].rolling(9).min()) / 2.0
    base = (df["high"].rolling(26).max() + df["low"].rolling(26).min()) / 2.0
    span_a = ((conv + base) / 2.0).shift(26)
    span_b = ((df["high"].rolling(52).max() + df["low"].rolling(52).min()) / 2.0).shift(26)
    df["ichimoku_conv_9"] = conv
    df["ichimoku_base_26"] = base
    df["ichimoku_span_a"] = span_a
    df["ichimoku_span_b"] = span_b
    return ["ichimoku_conv_9", "ichimoku_base_26", "ichimoku_span_a", "ichimoku_span_b"]


FEATURE_BUILDERS.update({
    "VWAP": add_vwap_features,
    "MOMENTUM": add_momentum_features,
    "ORDER_FLOW": add_orderflow_features,
    "ZSCORE": add_zscore_features,
    "DONCHIAN": add_donchian_features,
    "STOCHASTIC": add_stochastic_features,
    "KELTNER": add_keltner_features,
    "ADX": add_adx_features,
    "CCI": add_cci_features,
    "WILLIAMS_R": add_williams_r_features,
    "OBV": add_obv_features,
    "CMF": add_cmf_features,
    "ICHIMOKU": add_ichimoku_features,
})

def generate_features(df: pd.DataFrame, selected_features: list[str]):
    out = ensure_sorted(df.copy())
    generated_cols: list[str] = []
    for feature_name in selected_features:
        builder = FEATURE_BUILDERS.get(feature_name.upper())
        if builder is None:
            continue
        generated_cols.extend(builder(out))
    return out, generated_cols
