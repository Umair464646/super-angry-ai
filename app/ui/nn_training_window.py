from __future__ import annotations

from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QTextEdit
import pyqtgraph as pg


class NNTrainingWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Neural Network Training")
        self.resize(900, 620)

        self.loss_x = []
        self.loss_y = []
        self.acc_x = []
        self.acc_y = []

        layout = QVBoxLayout(self)
        self.arch_label = QLabel("Architecture: waiting...")
        self.arch_label.setStyleSheet("font-size:15px; font-weight:700;")

        self.loss_plot = pg.PlotWidget(title="NN Loss")
        self.loss_plot.setLabel("left", "Loss")
        self.loss_plot.setLabel("bottom", "Epoch")

        self.acc_plot = pg.PlotWidget(title="NN Accuracy")
        self.acc_plot.setLabel("left", "Accuracy")
        self.acc_plot.setLabel("bottom", "Epoch")

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        layout.addWidget(self.arch_label)
        layout.addWidget(self.loss_plot, 2)
        layout.addWidget(self.acc_plot, 2)
        layout.addWidget(self.log_box, 2)

    def set_architecture(self, text: str):
        self.arch_label.setText(f"Architecture: {text}")

    def on_epoch(self, epoch: int, total: int, loss: float, acc: float):
        self.loss_x.append(epoch)
        self.loss_y.append(loss)
        self.acc_x.append(epoch)
        self.acc_y.append(acc)

        self.loss_plot.clear()
        self.acc_plot.clear()
        self.loss_plot.plot(self.loss_x, self.loss_y, pen=pg.mkPen("#ff6b6b", width=2))
        self.acc_plot.plot(self.acc_x, self.acc_y, pen=pg.mkPen("#00d4ff", width=2))
        self.log_box.append(f"Epoch {epoch}/{total} | loss={loss:.5f} | acc={acc:.4f}")

    def on_finished(self):
        self.log_box.append("Training complete.")
