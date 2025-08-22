from pathlib import Path
import json
import csv
import time
from importlib import metadata
from importlib.resources import files
import random
import subprocess

HOME_DIR = Path.home() / ".PyResBench"
HOME_DIR.mkdir(parents=True, exist_ok=True)

def now():
    return time.perf_counter()

def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def append_history_csv(row_dict):
    path = HOME_DIR / "history.csv"
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row_dict)

def load_ascii_art() -> str:
    # Пытаемся взять из пакетных данных
    try:
        p = files("PyResBench") / "PyResBench.txt"
        return p.read_text(encoding="utf-8")
    except Exception:
        pass
    # Фоллбэк: поиск рядом с проектом (на случай dev-режима)
    for up in [Path(__file__).resolve().parents[i] for i in range(1,6)]:
        cand = up / "PyResBench.txt"
        if cand.exists():
            return cand.read_text(encoding="utf-8")
    return "PyResBench"

def get_version_fallback() -> str:
    try:
        return metadata.version("PyResBench")
    except Exception:
        from . import __version__
        return getattr(__version__, "__version__", "0.0.0")

def get_git_short_hash() -> str | None:
    # Опционально: короткий хэш, если рядом есть .git
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, timeout=2)
        return out.strip()
    except Exception:
        return None

TIPS = [
    "Tip: используйте --dataset synthetic для быстрого сухого прогона.",
    "Tip: --no-amp отключит смешанную точность на CUDA.",
    "Tip: записать JSON и CSV: --json-out run.json --csv-out run.csv.",
    "Tip: увеличьте --workers для ускорения даталоадера (но следите за RAM).",
    "Tip: на CPU ставьте меньший --batch-size (например, 32/64).",
]

def random_tip() -> str:
    return random.choice(TIPS)