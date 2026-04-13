from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QWidget,
    QHBoxLayout,
    QFrame,
    QTabWidget,
)
from PyQt6.QtCore import QTimer
import pyqtgraph as pg
import math


class AILiveMonitorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Research Live Monitor")
        self.resize(1540, 920)

        self.fitness_points_x = []
        self.fitness_points_y = []
        self.candidate_points_x = []
        self.candidate_points_y = []
        self.pulse_x = list(range(140))
        self.pulse_phase = 0.0

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        hero = QFrame()
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(14, 12, 14, 12)

        left_block = QVBoxLayout()
        self.stage_label = QLabel("Stage: idle")
        self.stage_label.setObjectName("HeroTitle")
        self.sub_label = QLabel("Pipeline waiting for Start. Open this monitor before or during runs.")
        self.sub_label.setObjectName("HeroSub")
        left_block.addWidget(self.stage_label)
        left_block.addWidget(self.sub_label)

        right_block = QHBoxLayout()
        self.chip_candidate = QLabel("Candidate tests: idle")
        self.chip_candidate.setObjectName("Chip")
        self.chip_overall = QLabel("Overall: 0%")
        self.chip_overall.setObjectName("Chip")
        right_block.addWidget(self.chip_candidate)
        right_block.addWidget(self.chip_overall)

        hero_layout.addLayout(left_block, 3)
        hero_layout.addLayout(right_block, 2)

        self.overall_progress = QProgressBar()
        self.overall_progress.setRange(0, 100)
        self.overall_progress.setTextVisible(True)
        self.candidate_progress = QProgressBar()
        self.candidate_progress.setRange(0, 100)
        self.candidate_progress.setTextVisible(True)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)

        pipeline_tab = QWidget()
        pipeline_layout = QHBoxLayout(pipeline_tab)
        pipeline_layout.setContentsMargins(0, 0, 0, 0)
        pipeline_layout.setSpacing(10)

        left_panel = QFrame()
        lp = QVBoxLayout(left_panel)
        lp.setContentsMargins(10, 10, 10, 10)
        lp.addWidget(QLabel("Pipeline Timeline"))
        self.timeline_table = QTableWidget(0, 3)
        self.timeline_table.setHorizontalHeaderLabels(["Stage", "%", "Note"])
        self.timeline_table.setAlternatingRowColors(True)
        lp.addWidget(self.timeline_table, 2)
        lp.addWidget(QLabel("Generation Summary"))
        self.gen_table = QTableWidget(0, 4)
        self.gen_table.setHorizontalHeaderLabels(["Gen", "Population", "Survivors", "Best Fitness"])
        self.gen_table.setAlternatingRowColors(True)
        lp.addWidget(self.gen_table, 1)

        right_panel = QFrame()
        rp = QVBoxLayout(right_panel)
        rp.setContentsMargins(10, 10, 10, 10)
        self.fitness_plot = pg.PlotWidget(title="Best Fitness by Generation")
        self.fitness_plot.setLabel("left", "Fitness")
        self.fitness_plot.setLabel("bottom", "Generation")
        self.candidate_plot = pg.PlotWidget(title="Candidate Test Throughput")
        self.candidate_plot.setLabel("left", "Done / Total")
        self.candidate_plot.setLabel("bottom", "Tick")
        self.pulse_plot = pg.PlotWidget(title="AI Pulse")
        self.pulse_plot.setLabel("left", "Activity")
        self.pulse_plot.setLabel("bottom", "Frame")
        rp.addWidget(self.fitness_plot, 2)
        rp.addWidget(self.candidate_plot, 2)
        rp.addWidget(self.pulse_plot, 2)

        pipeline_layout.addWidget(left_panel, 2)
        pipeline_layout.addWidget(right_panel, 3)

        logs_tab = QWidget()
        logs_layout = QVBoxLayout(logs_tab)
        logs_layout.setContentsMargins(0, 0, 0, 0)
        logs_layout.addWidget(QLabel("Live Engine Logs"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        logs_layout.addWidget(self.log_box)

        tabs.addTab(pipeline_tab, "Pipeline")
        tabs.addTab(logs_tab, "Logs")

        root.addWidget(hero)
        root.addWidget(self.overall_progress)
        root.addWidget(self.candidate_progress)
        root.addWidget(tabs, 1)

        self._apply_theme()

        self.pulse_timer = QTimer(self)
        self.pulse_timer.setInterval(120)
        self.pulse_timer.timeout.connect(self._tick_pulse)
        self.pulse_timer.start()

    def _apply_theme(self):
        self.setStyleSheet(
            """
            QDialog { background: #060b12; color: #eaf2ff; }
            QFrame { background: #0a131f; border: 1px solid #1a2b3f; border-radius: 10px; }
            QLabel#HeroTitle { font-size: 20px; font-weight: 700; color: #f4fbff; }
            QLabel#HeroSub { color: #9fb6cc; }
            QLabel#Chip { background: #0f1f30; border: 1px solid #2a3d56; border-radius: 12px; padding: 6px 12px; color: #91dcff; font-weight: 600; }
            QProgressBar { border: 1px solid #22344a; border-radius: 8px; background: #08121d; text-align: center; min-height: 20px; }
            QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00b8ff, stop:1 #00e0b8); border-radius: 8px; }
            QTableWidget { background: #09111b; border: 1px solid #1c2a3a; alternate-background-color: #0d1624; gridline-color: #1c2a3a; }
            QHeaderView::section { background: #0f1a28; color: #9fb5cc; padding: 6px; border: 1px solid #1b2b3d; font-weight: 600; }
            QTextEdit { background: #09111b; border: 1px solid #1c2a3a; color: #d6e5f5; }
            QTabWidget::pane { border: 1px solid #1a2b3f; border-radius: 8px; background: #07101a; }
            QTabBar::tab { background: #0d1624; color: #9fb5cc; border: 1px solid #1a2b3f; padding: 8px 12px; border-top-left-radius: 8px; border-top-right-radius: 8px; }
            QTabBar::tab:selected { background: #112237; color: #eaf2ff; }
            """
        )

    def on_progress(self, value: int):
        v = max(0, min(100, int(value)))
        self.overall_progress.setValue(v)
        self.chip_overall.setText(f"Overall: {v}%")

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
        self.chip_candidate.setText(f"Candidate tests | gen {gen} | {family} | {done}/{total}")
        self.candidate_points_x.append(len(self.candidate_points_x) + 1)
        self.candidate_points_y.append(done / max(1, total))
        self.candidate_plot.clear()
        self.candidate_plot.plot(
            self.candidate_points_x[-300:],
            self.candidate_points_y[-300:],
            pen=pg.mkPen(color="#00d4ff", width=2),
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
        self.chip_candidate.setText("Candidate tests: complete")
        self.log_box.append("[INFO] Live monitor: run complete")

    def _tick_pulse(self):
        self.pulse_phase += 0.25
        y = [0.5 + 0.45 * math.sin((i / 9.0) + self.pulse_phase) for i in self.pulse_x]
        self.pulse_plot.clear()
        self.pulse_plot.plot(self.pulse_x, y, pen=pg.mkPen(color="#ff9f1a", width=2))
