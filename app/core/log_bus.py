from datetime import datetime

def format_log_line(level: str, text: str) -> str:
    ts = datetime.now().strftime("%H:%M:%S")
    return f"[{ts}] [{level.upper()}] {text}"
