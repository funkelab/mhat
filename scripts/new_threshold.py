from darts_utils.segmentation import compute_segmentation
import csv
from pathlib import Path
import zarr
import numpy as np
from evaluate import evaluate_masks

if __name__ == "__main__":
    
    # thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    
    # for threshold in thresholds:

    #     total_FN = 0
    #     total_FP = 0
    #     total_TP = 0
    threshold = 0.15

    for file_num in range(1,101):

        base_path = Path(f'/nrs/funke/data/darts/synthetic_data/test1/{file_num}/data.zarr')
        root = zarr.open(base_path)
        fragments = root["fragments"][:]
        # gt_mask = root["mask"][:]
        merge_history = Path(base_path).parent / "merge_history.csv"
        segmentation = compute_segmentation(fragments, merge_history, threshold)

        root["pred_mask_0.15"] = segmentation
        
        # evaluate_masks(base_path, gt_mask, segmentation)

        #     csv_file = base_path.parent / "new_evaluation.csv"
        #     with open(csv_file, "r") as f:
        #         datareader = csv.reader(f)
        #         next(datareader)
        #         for r in datareader:
        #             total_TP += int(r[1])
        #             total_FP += int(r[2])
        #             total_FN += int(r[3])
        
        # print(total_TP)
        # print(total_FP)
        # print(total_FN)
        # precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
        # recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
        # f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # # find a better way to save the different f1 scores
        # print(f'The F1 Score for Threshold {threshold} is', f1)