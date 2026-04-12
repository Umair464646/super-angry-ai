from __future__ import annotations

from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QTextEdit, QHBoxLayout, QTabWidget, QWidget, QGridLayout
from PyQt6.QtCore import Qt
import pyqtgraph as pg
import numpy as np


class NNTrainingWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Neural Network Training")
        self.resize(1080, 760)

        self.loss_x = []
        self.loss_y = []
        self.acc_x = []
        self.acc_y = []
        self.val_loss_y = []
        self.val_acc_y = []
        self.prec_y = []
        self.rec_y = []
        self.f1_y = []
        self.lr_y = []
        self.grad_y = []
        self.drift_y = []
        self.current_generation = 0
        self.candidate_note = "waiting..."

        layout = QVBoxLayout(self)
        self.arch_label = QLabel("Architecture: waiting...")
        self.arch_label.setStyleSheet("font-size:15px; font-weight:700;")
        self.status_label = QLabel("Pipeline generation: waiting...")
        self.status_label.setStyleSheet("color:#8fb0d4; font-size:13px;")
        self.sample_label = QLabel("Current sample: waiting | Prediction: n/a | Probability: n/a | Confidence: n/a")
        self.sample_label.setStyleSheet("color:#9cc3e8; font-size:12px;")

        chip_row = QHBoxLayout()
        self.loss_chip = QLabel("Latest loss: -")
        self.acc_chip = QLabel("Latest acc: -")
        self.epoch_chip = QLabel("Epoch: -")
        for chip in (self.loss_chip, self.acc_chip, self.epoch_chip):
            chip.setStyleSheet("padding:4px 8px; border:1px solid #2a3d56; border-radius:10px;")
            chip_row.addWidget(chip)
        chip_row.addStretch(1)

        self.param_label = QLabel("Params: total=0 | trainable=0")
        self.param_label.setStyleSheet("color:#8fb0d4; font-size:12px;")
        self.layer_sizes: list[int] = []

        self.topology_plot = pg.PlotWidget(title="NN Topology")
        self.topology_plot.setMouseEnabled(x=False, y=False)
        self.topology_plot.hideAxis("left")
        self.topology_plot.hideAxis("bottom")
        self.topology_plot.setYRange(-1, 1)

        self.loss_plot = pg.PlotWidget(title="NN Loss")
        self.loss_plot.setLabel("left", "Loss")
        self.loss_plot.setLabel("bottom", "Epoch")

        self.acc_plot = pg.PlotWidget(title="NN Accuracy")
        self.acc_plot.setLabel("left", "Accuracy")
        self.acc_plot.setLabel("bottom", "Epoch")

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        self.feature_bar = pg.PlotWidget(title="Feature Importance")
        self.feature_bar.setLabel("left", "Importance")
        self.feature_bar.setLabel("bottom", "Feature")
        self.activation_map = pg.ImageView(view=pg.PlotItem())
        self.activation_map.ui.histogram.hide()
        self.activation_map.ui.roiBtn.hide()
        self.activation_map.ui.menuBtn.hide()
        self.output_meter = pg.PlotWidget(title="Output Confidence")
        self.output_meter.setYRange(0.0, 1.0)
        self.output_meter.setLabel("left", "Confidence")
        self.output_meter.setLabel("bottom", "Epoch")
        self.regime_plot = pg.PlotWidget(title="Regime Probabilities")
        self.regime_plot.setYRange(0.0, 1.0)
        self.regime_plot.setLabel("left", "P")
        self.regime_plot.setLabel("bottom", "Epoch")

        tabs = QTabWidget()
        arch_tab = QWidget()
        arch_layout = QVBoxLayout(arch_tab)
        arch_layout.addWidget(self.topology_plot, 4)
        arch_layout.addWidget(self.param_label)

        train_tab = QWidget()
        train_layout = QVBoxLayout(train_tab)
        train_layout.addWidget(self.loss_plot, 2)
        train_layout.addWidget(self.acc_plot, 2)

        metrics_tab = QWidget()
        metrics_grid = QGridLayout(metrics_tab)
        self.prec_plot = pg.PlotWidget(title="Precision / Recall / F1")
        self.prec_plot.setLabel("left", "Score")
        self.prec_plot.setLabel("bottom", "Epoch")
        self.grad_plot = pg.PlotWidget(title="Learning Rate / Gradient Norm")
        self.grad_plot.setLabel("left", "Value")
        self.grad_plot.setLabel("bottom", "Epoch")
        self.drift_plot = pg.PlotWidget(title="Drift / Degradation")
        self.drift_plot.setLabel("left", "Drift")
        self.drift_plot.setLabel("bottom", "Epoch")
        metrics_grid.addWidget(self.prec_plot, 0, 0)
        metrics_grid.addWidget(self.grad_plot, 0, 1)
        metrics_grid.addWidget(self.drift_plot, 1, 0, 1, 2)

        intel_tab = QWidget()
        intel_grid = QGridLayout(intel_tab)
        intel_grid.addWidget(self.feature_bar, 0, 0)
        intel_grid.addWidget(self.output_meter, 0, 1)
        intel_grid.addWidget(self.regime_plot, 1, 0)
        intel_grid.addWidget(self.activation_map, 1, 1)

        tabs.addTab(arch_tab, "Architecture")
        tabs.addTab(train_tab, "Training")
        tabs.addTab(metrics_tab, "Metrics")
        tabs.addTab(intel_tab, "Intelligence")

        layout.addWidget(self.arch_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.sample_label)
        layout.addLayout(chip_row)
        layout.addWidget(tabs, 6)
        layout.addWidget(self.log_box, 2)

    def set_architecture(self, text: str):
        self.arch_label.setText(f"Architecture: {text}")
        self._draw_topology(text)

    def on_generation(self, gen: int, survivors: int, population: int):
        self.current_generation = int(gen)
        self.status_label.setText(
            f"Pipeline generation: {gen} | survivors={survivors} | population={population} | candidate={self.candidate_note}"
        )

    def on_candidate(self, gen: int, done: int, total: int, family: str):
        self.candidate_note = f"{family} {done}/{total}"
        if self.current_generation <= 0:
            self.current_generation = int(gen)
        self.status_label.setText(
            f"Pipeline generation: {self.current_generation} | candidate={family} {done}/{total}"
        )

    def on_epoch(self, epoch: int, total: int, loss: float, acc: float, extra: dict | None = None):
        extra = extra or {}
        self.loss_x.append(epoch)
        self.loss_y.append(loss)
        self.acc_x.append(epoch)
        self.acc_y.append(acc)
        self.val_loss_y.append(float(extra.get("val_loss", loss)))
        self.val_acc_y.append(float(extra.get("val_acc", acc)))
        self.prec_y.append(float(extra.get("precision", 0.0)))
        self.rec_y.append(float(extra.get("recall", 0.0)))
        self.f1_y.append(float(extra.get("f1", 0.0)))
        self.lr_y.append(float(extra.get("lr", 0.0)))
        self.grad_y.append(float(extra.get("grad_norm", 0.0)))
        self.drift_y.append(float(extra.get("drift", 0.0)))

        self.loss_plot.clear()
        self.acc_plot.clear()
        self.loss_plot.plot(self.loss_x, self.loss_y, pen=pg.mkPen("#ff6b6b", width=2))
        self.loss_plot.plot(self.loss_x, self.val_loss_y, pen=pg.mkPen("#ffadad", width=2, style=Qt.PenStyle.DashLine))
        self.acc_plot.plot(self.acc_x, self.acc_y, pen=pg.mkPen("#00d4ff", width=2))
        self.acc_plot.plot(self.acc_x, self.val_acc_y, pen=pg.mkPen("#8ee6ff", width=2, style=Qt.PenStyle.DashLine))
        self.prec_plot.clear()
        self.prec_plot.plot(self.loss_x, self.prec_y, pen=pg.mkPen("#ffd166", width=2), name="precision")
        self.prec_plot.plot(self.loss_x, self.rec_y, pen=pg.mkPen("#06d6a0", width=2), name="recall")
        self.prec_plot.plot(self.loss_x, self.f1_y, pen=pg.mkPen("#ef476f", width=2), name="f1")
        self.grad_plot.clear()
        self.grad_plot.plot(self.loss_x, self.lr_y, pen=pg.mkPen("#bdb2ff", width=2))
        self.grad_plot.plot(self.loss_x, self.grad_y, pen=pg.mkPen("#9bf6ff", width=2))
        self.drift_plot.clear()
        self.drift_plot.plot(self.loss_x, self.drift_y, pen=pg.mkPen("#ff8fab", width=2))

        confidence = float(extra.get("output_confidence", 0.0))
        sample = str(extra.get("current_sample", "n/a"))
        pred = str(extra.get("prediction", "n/a"))
        prob = float(extra.get("probability", 0.0))
        self.sample_label.setText(
            f"Current sample: {sample} | Prediction: {pred} | Probability: {prob:.2f} | Confidence: {confidence:.2f}"
        )
        self.output_meter.clear()
        self.output_meter.plot(self.loss_x, [confidence] * len(self.loss_x), pen=pg.mkPen("#7bffb1", width=2))
        self.output_meter.addItem(pg.InfiniteLine(pos=0.5, angle=0, pen=pg.mkPen("#555", width=1)))

        layer_activity = extra.get("layer_activity", [])
        if layer_activity:
            mat = np.array([layer_activity], dtype=float)
            self.activation_map.setImage(mat, autoLevels=True)
        feature_strength = extra.get("feature_strength", [])
        if feature_strength:
            y = np.array(feature_strength, dtype=float)
            x = np.arange(len(y))
            bg = pg.BarGraphItem(x=x, height=y, width=0.7, brush="#4cc9f0")
            self.feature_bar.clear()
            self.feature_bar.addItem(bg)
            self._draw_topology(self.arch_label.text().replace("Architecture: ", ""), layer_activity=layer_activity, feature_strength=y, output_conf=confidence)

        rp = float(extra.get("val_acc", 0.0))
        self.regime_plot.clear()
        self.regime_plot.plot(self.loss_x, [rp] * len(self.loss_x), pen=pg.mkPen("#f8961e", width=2))

        self.loss_chip.setText(f"Latest loss: {loss:.5f}")
        self.acc_chip.setText(f"Latest acc: {acc:.4f}")
        self.epoch_chip.setText(f"Epoch: {epoch}/{total}")
        self.log_box.append(f"Epoch {epoch}/{total} | loss={loss:.5f} | acc={acc:.4f}")

    def on_finished(self):
        self.log_box.append("Training complete.")

    def reset_run(self):
        self.loss_x.clear()
        self.loss_y.clear()
        self.acc_x.clear()
        self.acc_y.clear()
        self.current_generation = 0
        self.candidate_note = "waiting..."
        self.loss_plot.clear()
        self.acc_plot.clear()
        self.prec_plot.clear()
        self.grad_plot.clear()
        self.drift_plot.clear()
        self.feature_bar.clear()
        self.output_meter.clear()
        self.regime_plot.clear()
        self.log_box.clear()
        self.status_label.setText("Pipeline generation: waiting...")
        self.sample_label.setText("Current sample: waiting | Prediction: n/a | Probability: n/a | Confidence: n/a")
        self.loss_chip.setText("Latest loss: -")
        self.acc_chip.setText("Latest acc: -")
        self.epoch_chip.setText("Epoch: -")

    def _draw_topology(self, arch: str, layer_activity: list[float] | None = None, feature_strength=None, output_conf: float = 0.5):
        self.topology_plot.clear()
        if not arch:
            return
        layer_sizes: list[int] = []
        for part in [x.strip() for x in arch.split("->")]:
            if "(" not in part or ")" not in part:
                continue
            try:
                val = int(part.split("(", 1)[1].split(")", 1)[0].split(",")[0].strip())
                layer_sizes.append(max(1, min(14, val)))
            except Exception:
                continue
        if len(layer_sizes) < 2:
            return
        self.layer_sizes = layer_sizes
        layer_activity = layer_activity or [0.4] * len(layer_sizes)

        x_step = 1.0 / max(1, len(layer_sizes) - 1)
        layer_points: list[list[tuple[float, float]]] = []
        for idx, size in enumerate(layer_sizes):
            x = idx * x_step
            ys = [0.0] if size == 1 else [1 - (2 * i / (size - 1)) for i in range(size)]
            points = [(x, y * 0.8) for y in ys]
            layer_points.append(points)

        total_params = 0
        for i in range(len(layer_points) - 1):
            left = layer_points[i]
            right = layer_points[i + 1]
            total_params += len(left) * len(right)
            edge_weight = 1.0
            if feature_strength is not None and i == 0 and len(feature_strength) > 0:
                edge_weight = float(np.clip(np.mean(feature_strength), 0.2, 3.0))
            for x1, y1 in left:
                for x2, y2 in right:
                    self.topology_plot.plot(
                        [x1, x2],
                        [y1, y2],
                        pen=pg.mkPen(color=(70, 110, 160, min(255, int(70 + 80 * edge_weight))), width=max(1, int(edge_weight))),
                    )
        for li, pts in enumerate(layer_points):
            x = [p[0] for p in pts]
            y = [p[1] for p in pts]
            act = float(np.clip(layer_activity[min(li, len(layer_activity) - 1)], 0.0, 1.2))
            if li == len(layer_points) - 1:
                if output_conf > 0.66:
                    brush = (70, 250, 160, min(255, int(120 + 120 * output_conf)))
                elif output_conf < 0.34:
                    brush = (255, 120, 120, min(255, int(120 + 120 * (1 - output_conf))))
                else:
                    brush = (255, 205, 86, 190)
            else:
                v = min(255, int(80 + act * 160))
                brush = (v, 220, 255, min(255, int(130 + act * 80)))
            self.topology_plot.plot(
                x,
                y,
                pen=None,
                symbol="o",
                symbolSize=max(7, 7 + int(act * 5)),
                symbolBrush=brush,
                symbolPen=pg.mkPen("#63d8ff", width=1),
            )
        self.param_label.setText(f"Params: total={total_params:,} | trainable={total_params:,}")
