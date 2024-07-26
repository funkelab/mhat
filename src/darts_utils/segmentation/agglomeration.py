import csv
from pathlib import Path
import zarr
import numpy as np
from evaluate.py import evaluate_masks

def compute_segmentation(fragments: np.ndarray, merge_history: Path, threshold: float):
    """Takes fragments and a merge history, and merges the fragments up until the threshold.
    Returns a new numpy array that is a "final" segmentation.

    Args:
        fragments (np.ndarray): Fragments that have already been run through waterz
        merge_history (Path): Result of running waterz on fragments with pretty high threshold
        threshold (float): The score threshold to merge until. If it is bigger than 
            the largest in the merge history, nothing more will merge.

    Returns:
        np.ndarray: A segmentation coming from merging the fragments according to the merge history
    """
    with open(merge_history) as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = int(row["a"])
            b = int(row["b"])
            c = int(row["c"])
            score = float(row["score"])
            if score <= threshold:
                fragments[fragments == a] = c
                fragments[fragments == b] = c

    return fragments

if __name__ == "__main__":
    
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    
    for threshold in thresholds:

        total_FN = 0
        total_FP = 0
        total_TP = 0

        for file_num in range(1,11):

            base_path = Path(f'/nrs/funke/data/darts/synthetic_data/validation1/{file_num}/data.zarr')
            root = zarr.open(base_path)
            fragments = root["fragments"][:]
            gt_mask = root["mask"][:]
            merge_history = Path(base_path).parent / "merge_history.csv"
            segmentation = compute_segmentation(fragments, merge_history, threshold)
            
            evaluate_masks(base_path, gt_mask, segmentation)

            csv_file = root.parent / "new_evaluation.csv"
            with open(csv_file) as f:
                total_TP = sum(int(r[1]) for r in csv.DictReader(f))
                total_FP = sum(int(r[2]) for r in csv.DictReader(f))
                total_FN = sum(int(r[3]) for r in csv.DictReader(f))
        
        precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
        recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # find a better way to save the different f1 scores
        print(f'The F1 Score for Threshold {threshold} is', f1)
