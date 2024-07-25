import csv
from pathlib import Path

import numpy as np


def compute_segmentation(fragments: np.ndarray, merge_history: Path, threshold: float):
    with open(merge_history) as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = int(row["a"])
            b = int(row["b"])
            c = int(row["c"])
            score = float(row["score"])
            if score >= threshold:
                fragments[fragments == a] = c
                fragments[fragments == b] = c

    return fragments
