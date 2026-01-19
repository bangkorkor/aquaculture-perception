import csv
from pathlib import Path
from typing import Dict, Any


def load_runs_csv(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    runs: Dict[str, Dict[str, Any]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = (row.get("id") or "").strip()
            if not rid:
                continue
            runs[rid] = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
    return runs
