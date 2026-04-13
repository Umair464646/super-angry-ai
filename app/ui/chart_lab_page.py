from __future__ import annotations
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox, QPushButton

pg.setConfigOptions(antialias=False, background="#0d1117", foreground="#c9d1d9")

class TimeAxisItem(pg.AxisItem):
    def __init__(self, orientation="bottom"):
        super().__init__(orientation=orientation)
        self._timestamps = []
    def set_timestamps(self, timestamps):
        self._timestamps = list(timestamps)
    def tickStrings(self, values, scale, spacing):
        labels = []
        n = len(self._timestamps)
        for value in values:
            idx = int(round(value))
            if 0 <= idx < n:
                ts = self._timestamps[idx]
                try:
                    labels.append(pd.Timestamp(ts).strftime("%m-%d %H:%M"))
                except Exception:
                    labels.append(str(ts))
            else:
                labels.append("")
        return labels

class PriceAxisItem(pg.AxisItem):
    def __init__(self, orientation="left"):
        super().__init__(orientation=orientation)
        self.decimals = 6
    def set_prices(self, prices):
        arr = np.asarray(prices, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            self.decimals = 6
            return
        median_price = float(np.median(np.abs(arr)))
        if median_price >= 1000: self.decimals = 2
        elif median_price >= 100: self.decimals = 3
        elif median_price >= 1: self.decimals = 4
        elif median_price >= 0.01: self.decimals = 6
        else: self.decimals = 8
    def tickStrings(self, values, scale, spacing):
        fmt = "{:." + str(self.decimals) + "f}"
        return [fmt.format(v) for v in values]

class ChartLabPage(QWidget):
    timeframe_requested = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.cache = {}
        self.current_df = None
        self._wick_items = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        title = QLabel("Chart Lab")
        title.setStyleSheet("font-size: 24px; font-weight: 700;")
        subtitle = QLabel("Optimized chart view with explicit wicks, disk-cached timeframes, and chart window selector.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #8a95a5;")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        controls = QHBoxLayout()
        self.timeframe_box = QComboBox()
        self.timeframe_box.addItems(["1s","5s","15s","30s","1m","5m","15m","1h","4h"])
        self.timeframe_box.setCurrentText("1m")
        self.timeframe_box.currentTextChanged.connect(self.on_timeframe_changed)

        self.mode_box = QComboBox()
        self.mode_box.addItems(["Candles","Line"])
        self.mode_box.setCurrentText("Candles")
        self.mode_box.currentTextChanged.connect(self.render_chart)

        self.window_box = QComboBox()
        self.window_box.addItems(["500","1500","5000","Full"])
        self.window_box.setCurrentText("1500")
        self.window_box.currentTextChanged.connect(self.render_chart)

        self.hide_zero = QCheckBox("Hide zero-volume bars")
        self.hide_zero.setChecked(True)
        self.hide_zero.stateChanged.connect(self.render_chart)

        self.hide_synth = QCheckBox("Hide synthetic rows")
        self.hide_synth.setChecked(False)
        self.hide_synth.stateChanged.connect(self.render_chart)

        self.reset_btn = QPushButton("Reset View")
        self.reset_btn.clicked.connect(self.reset_view)

        self.range_label = QLabel("Range: —")
        self.range_label.setStyleSheet("color: #8a95a5;")
        self.cursor_label = QLabel("Cursor: —")
        self.cursor_label.setStyleSheet("color: #8a95a5;")

        controls.addWidget(QLabel("Timeframe")); controls.addWidget(self.timeframe_box)
        controls.addWidget(QLabel("Mode")); controls.addWidget(self.mode_box)
        controls.addWidget(QLabel("Bars")); controls.addWidget(self.window_box)
        controls.addWidget(self.hide_zero); controls.addWidget(self.hide_synth); controls.addWidget(self.reset_btn)
        controls.addStretch(1)
        layout.addLayout(controls)
        layout.addWidget(self.range_label)
        layout.addWidget(self.cursor_label)

        self.graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics, 1)

        self.time_axis_top = TimeAxisItem("bottom")
        self.time_axis_bottom = TimeAxisItem("bottom")
        self.price_axis = PriceAxisItem("left")

        self.price_plot = self.graphics.addPlot(row=0, col=0, axisItems={"bottom": self.time_axis_top, "left": self.price_axis})
        self.volume_plot = self.graphics.addPlot(row=1, col=0, axisItems={"bottom": self.time_axis_bottom})
        for plot in [self.price_plot, self.volume_plot]:
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.setMenuEnabled(False)
            plot.disableAutoRange()

        self.price_plot.setXLink(self.volume_plot)
        self.price_plot.setLabel("left", "Price")
        self.volume_plot.setLabel("left", "Volume")
        self.volume_plot.setMaximumHeight(220)
        self.volume_plot.setMinimumHeight(160)

        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#6e7681", width=1))
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#6e7681", width=1))
        self.price_plot.addItem(self.vline, ignoreBounds=True)
        self.price_plot.addItem(self.hline, ignoreBounds=True)
        self.mouse_proxy = pg.SignalProxy(self.price_plot.scene().sigMouseMoved, rateLimit=60, slot=self._mouse_moved)

    def set_base_dataset(self, df):
        self.cache = {"1s": df.copy()}
        self.on_timeframe_changed(self.timeframe_box.currentText())

    def set_timeframe_dataset(self, timeframe, df):
        self.cache[timeframe] = df
        if timeframe == self.timeframe_box.currentText():
            self.render_chart()

    def on_timeframe_changed(self, timeframe):
        if timeframe not in self.cache:
            self.timeframe_requested.emit(timeframe)
            return
        self.render_chart()

    def _get_filtered_df(self):
        tf = self.timeframe_box.currentText()
        df = self.cache.get(tf)
        if df is None or df.empty:
            return None
        local = df
        if self.hide_zero.isChecked() and "volume" in local.columns:
            local = local[local["volume"] > 0]
        if self.hide_synth.isChecked() and "synthetic" in local.columns:
            local = local[local["synthetic"] != 1]
        if local.empty:
            return None
        bars_value = self.window_box.currentText()
        if bars_value != "Full":
            limit = int(bars_value)
            if len(local) > limit:
                local = local.tail(limit)
        local = local.reset_index(drop=True).copy()
        local["bar_index"] = np.arange(len(local))
        return local

    def _plot_candles(self, df):
        x = df["bar_index"].to_numpy(dtype=float)
        opens = df["open"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        bullish = closes >= opens
        bearish = ~bullish
        for i in range(len(x)):
            color = "#00e676" if closes[i] >= opens[i] else "#ff4d4f"
            wick = pg.PlotDataItem([x[i], x[i]], [lows[i], highs[i]], pen=pg.mkPen(color, width=1.6))
            self.price_plot.addItem(wick)
            self._wick_items.append(wick)
        median_close = float(np.nanmedian(closes)) if len(closes) else 1.0
        min_body = max(abs(median_close) * 0.00005, 1e-8)
        bull_center = (opens[bullish] + closes[bullish]) / 2.0
        bull_height = np.maximum(np.abs(closes[bullish] - opens[bullish]), min_body)
        bear_center = (opens[bearish] + closes[bearish]) / 2.0
        bear_height = np.maximum(np.abs(closes[bearish] - opens[bearish]), min_body)
        if len(bull_center):
            self.price_plot.addItem(pg.BarGraphItem(x=x[bullish], y0=bull_center - (bull_height / 2.0), y1=bull_center + (bull_height / 2.0), width=0.62, brush="#00e676", pen="#00e676"))
        if len(bear_center):
            self.price_plot.addItem(pg.BarGraphItem(x=x[bearish], y0=bear_center - (bear_height / 2.0), y1=bear_center + (bear_height / 2.0), width=0.62, brush="#ff4d4f", pen="#ff4d4f"))

    def render_chart(self):
        self.price_plot.clear(); self.volume_plot.clear(); self._wick_items = []
        self.price_plot.addItem(self.vline, ignoreBounds=True); self.price_plot.addItem(self.hline, ignoreBounds=True)
        df = self._get_filtered_df()
        self.current_df = df
        if df is None or df.empty:
            label = pg.TextItem("No data for current chart filters", color="#8a95a5")
            self.price_plot.addItem(label); label.setPos(0, 0)
            self.range_label.setText("Range: —")
            return
        self.time_axis_top.set_timestamps(df["timestamp"].tolist())
        self.time_axis_bottom.set_timestamps(df["timestamp"].tolist())
        self.price_axis.set_prices(df["close"].to_numpy(dtype=float))
        x = df["bar_index"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        volumes = df["volume"].fillna(0).to_numpy(dtype=float)
        if self.mode_box.currentText() == "Line":
            self.price_plot.plot(x, closes, pen=pg.mkPen("#00d4ff", width=1.5))
        else:
            self._plot_candles(df)
        colors = ["#00e676" if c >= o else "#ff4d4f" for o, c in zip(df["open"], df["close"])]
        self.volume_plot.addItem(pg.BarGraphItem(x=x, height=volumes, width=0.7, brushes=colors, pens=colors))
        self.range_label.setText(f"Range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]} | visible bars {len(df):,}")
        self.reset_view()

    def reset_view(self):
        if self.current_df is None or self.current_df.empty:
            return
        df = self.current_df
        lows = df["low"].to_numpy(dtype=float); highs = df["high"].to_numpy(dtype=float); closes = df["close"].to_numpy(dtype=float); volumes = df["volume"].fillna(0).to_numpy(dtype=float)
        y_min = float(np.nanmin(lows)); y_max = float(np.nanmax(highs))
        y_range = max(y_max - y_min, 1e-12)
        price_ref = float(np.nanmedian(closes)) if len(closes) else 1.0
        y_pad = max(y_range * 0.15, abs(price_ref) * 0.0008)
        self.price_plot.setLimits(xMin=-5, xMax=max(len(df) + 5, 10))
        self.price_plot.setXRange(0, max(len(df) - 1, 10), padding=0)
        self.price_plot.setYRange(y_min - y_pad, y_max + y_pad, padding=0)
        vol_max = float(np.nanmax(volumes)) if len(volumes) else 1.0
        self.volume_plot.setLimits(xMin=-5, xMax=max(len(df) + 5, 10), yMin=0)
        self.volume_plot.setXRange(0, max(len(df) - 1, 10), padding=0)
        self.volume_plot.setYRange(0, vol_max * 1.15 if vol_max > 0 else 1, padding=0)

    def _mouse_moved(self, evt):
        if self.current_df is None or self.current_df.empty:
            return
        pos = evt[0]
        if not self.price_plot.sceneBoundingRect().contains(pos):
            return
        point = self.price_plot.vb.mapSceneToView(pos)
        x = int(round(point.x()))
        if x < 0 or x >= len(self.current_df):
            return
        row = self.current_df.iloc[x]
        self.vline.setPos(x); self.hline.setPos(point.y())
        synth = int(row["synthetic"]) if "synthetic" in row.index and pd.notna(row["synthetic"]) else 0
        self.cursor_label.setText(f"Cursor: {row['timestamp']} | O {row['open']:.6f} H {row['high']:.6f} L {row['low']:.6f} C {row['close']:.6f} | V {row['volume']:.3f} | synthetic {synth}")
