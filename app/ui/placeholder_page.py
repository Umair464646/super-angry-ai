from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

class PlaceholderPage(QWidget):
    def __init__(self, title: str, message: str):
        super().__init__()
        layout = QVBoxLayout(self)
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 24px; font-weight: 700;")
        body = QLabel(message)
        body.setWordWrap(True)
        body.setStyleSheet("color: #8a95a5; font-size: 14px;")
        layout.addWidget(title_label)
        layout.addWidget(body)
        layout.addStretch(1)
