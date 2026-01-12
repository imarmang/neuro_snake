# utils_logger.py
import csv
import os
from typing import Dict, Any


class CSVLogger:
    def __init__(self, filepath: str, fieldnames: list[str]):
        self.filepath = filepath
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Create file with header if missing
        if not os.path.exists(filepath):
            with open(filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        # Ensure only known columns are written
        filtered = {k: row.get(k, "") for k in self.fieldnames}
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(filtered)
