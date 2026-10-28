"""Collect the numbers a figure script quotes and save them to results/<fig>_numbers.json as {key: {value, note}}."""
import json

from lib.config import RESULTS_DIR


class FigureNumbers(dict):
    def __init__(self, fig):
        super().__init__()
        self.fig = fig

    def put(self, key, value, note=""):
        self[key] = {"value": value, "note": note}

    def save(self):
        path = RESULTS_DIR / f"{self.fig}_numbers.json"
        json.dump(self, open(path, "w"), indent=1)
        print(f"wrote {path.name}: {len(self)} numbers")
        return path
