from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QSplitter,
    QWidget,
    QHBoxLayout,
)
import pyqtgraph as pg


class AILiveMonitorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Research Live Monitor")
        self.resize(1400, 860)

        self.fitness_points_x = []
        self.fitness_points_y = []

        layout = QVBoxLayout(self)

        self.stage_label = QLabel("Stage: idle")
        self.stage_label.setStyleSheet("font-size:16px; font-weight:700;")

        self.overall_progress = QProgressBar()
        self.overall_progress.setRange(0, 100)

        self.candidate_progress_label = QLabel("Candidate tests: idle")
        self.candidate_progress = QProgressBar()
        self.candidate_progress.setRange(0, 100)

        top = QHBoxLayout()
        top.addWidget(self.stage_label, 2)
        top.addWidget(self.candidate_progress_label, 1)

        layout.addLayout(top)
        layout.addWidget(self.overall_progress)
        layout.addWidget(self.candidate_progress)

        split = QSplitter()

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)

        ll.addWidget(QLabel("Pipeline Timeline"))
        self.timeline_table = QTableWidget(0, 3)
        self.timeline_table.setHorizontalHeaderLabels(["Stage", "%", "Note"])
        ll.addWidget(self.timeline_table)

        ll.addWidget(QLabel("Generation Summary"))
        self.gen_table = QTableWidget(0, 4)
        self.gen_table.setHorizontalHeaderLabels(["Gen", "Population", "Survivors", "Best Fitness"])
        ll.addWidget(self.gen_table)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)

        self.fitness_plot = pg.PlotWidget(title="Best Fitness by Generation")
        self.fitness_plot.setLabel("left", "Fitness")
        self.fitness_plot.setLabel("bottom", "Generation")
        rl.addWidget(self.fitness_plot, 2)

        rl.addWidget(QLabel("Live Engine Logs"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        rl.addWidget(self.log_box, 3)

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)

        layout.addWidget(split, 1)

    def on_progress(self, value: int):
        self.overall_progress.setValue(max(0, min(100, int(value))))

    def on_stage(self, text: str):
        self.stage_label.setText(f"Stage: {text}")

    def on_timeline(self, stage_name: str, pct: int, note: str):
        r = self.timeline_table.rowCount()
        self.timeline_table.insertRow(r)
        self.timeline_table.setItem(r, 0, QTableWidgetItem(stage_name))
        self.timeline_table.setItem(r, 1, QTableWidgetItem(str(pct)))
        self.timeline_table.setItem(r, 2, QTableWidgetItem(note))
        self.timeline_table.scrollToBottom()

    def on_candidate(self, gen: int, done: int, total: int, family: str):
        pct = int((done / max(1, total)) * 100)
        self.candidate_progress.setValue(pct)
        self.candidate_progress_label.setText(
            f"Candidate tests | gen {gen} | {family} | {done}/{total}"
        )

    def on_generation(self, gen: int, survivors: int, best_fitness: float, population: int):
        r = self.gen_table.rowCount()
        self.gen_table.insertRow(r)
        self.gen_table.setItem(r, 0, QTableWidgetItem(str(gen)))
        self.gen_table.setItem(r, 1, QTableWidgetItem(str(population)))
        self.gen_table.setItem(r, 2, QTableWidgetItem(str(survivors)))
        self.gen_table.setItem(r, 3, QTableWidgetItem(f"{best_fitness:.2f}"))

        self.fitness_points_x.append(gen)
        self.fitness_points_y.append(best_fitness)
        self.fitness_plot.clear()
        self.fitness_plot.plot(
            self.fitness_points_x,
            self.fitness_points_y,
            pen=pg.mkPen(color="#7cfc00", width=2),
            symbol="o",
        )

    def on_log(self, level: str, message: str):
        self.log_box.append(f"[{level}] {message}")

    def on_finished(self):
        self.candidate_progress.setValue(100)
        self.candidate_progress_label.setText("Candidate tests: complete")
        self.log_box.append("[INFO] Live monitor: run complete")
