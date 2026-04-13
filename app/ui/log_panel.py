from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QPlainTextEdit
from PyQt6.QtGui import QTextCursor
from app.core.log_bus import format_log_line

class LogPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        title = QLabel("Logs")
        title.setStyleSheet("font-size: 16px; font-weight: 700;")
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear)
        top.addWidget(title)
        top.addStretch(1)
        top.addWidget(self.clear_btn)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(8000)
        layout.addLayout(top)
        layout.addWidget(self.text)

    def append(self, level: str, message: str):
        self.text.appendPlainText(format_log_line(level, message))
        cursor = self.text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.text.setTextCursor(cursor)

    def clear(self):
        self.text.clear()
