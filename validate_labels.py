import argparse
import pandas as pd
import numpy as np
from copy import copy
import importlib
import torch
from tqdm import tqdm

from src.data.utils import get_dataset, get_dataloader
from src.modules.metric import score, LABEL_COLS
from src.modules.utils import batch_to_device

def validate_perfect_maps_auc(val_dl, val_df, cfg):
    progress_bar = tqdm(val_dl, desc="Validate AUC labels")

    all_pseudo_preds = []
    all_series_uids = []

    with torch.no_grad():
        for batch in progress_bar:
            batch = batch_to_device(batch, cfg.device)

            target_maps = batch["target"]

            location_probs = torch.max(target_maps.view(target_maps.shape[0], 13, -1), dim=2).values
            present_prob = torch.max(location_probs, dim=1, keepdim=True).values
            final_probs = torch.cat([location_probs, present_prob], dim=1)
            
            all_pseudo_preds.append(final_probs.cpu().numpy())
            all_series_uids.extend(batch["series_uid"])

    y_pred = np.concatenate(all_pseudo_preds)
    ordered_df = val_df[val_df['SeriesInstanceUID'].isin(all_series_uids)].set_index('SeriesInstanceUID').loc[all_series_uids].reset_index()
    y_true = ordered_df[LABEL_COLS].values

    metrics = score(y_true, y_pred)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Validate ground-truth labels against the AUC metric.")
    parser.add_argument("--config", type=str, default="eval_config", help="Configuration file name")
    args = parser.parse_args()

    print("--- Starting label validation with AUC metric ---")

    config_path = args.config
    cfg = copy(importlib.import_module(f'src.configs.{args.config}').cfg)
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.local_rank = 0
    print(f"Configuration loaded: {config_path}")

    eval_df = pd.read_csv(cfg.eval_csv_path)
    eval_ds = get_dataset(eval_df, cfg, mode="val")

    cfg.batch_size_val = 1 
    eval_dl = get_dataloader(eval_ds, cfg, mode="val")
    print(f"Loaded {len(eval_ds)} samples for validation.")
    
    metrics = validate_perfect_maps_auc(eval_dl, eval_df, cfg)
    
    score_val = metrics['score']
    
    print("\n--- Label Validation Results ---")
    print(f"  Overall Score (weighted AUC): {score_val:.4f}")
    print(f"  'Aneurysm present' AUC:       {metrics['auc_present']:.4f}")
    print(f"  Mean location AUC:            {metrics['mean_location_auc']:.4f}")
    print("-------------------------------------------\n")

    if np.isclose(score_val, 1.0):
        print("SUCCESS: The result is 1.0, as expected.")
        print("Your labels and your metric pipeline are consistent. Any performance loss is due to the model.")
    else:
        print("WARNING: The score is NOT 1.0.")
        print("This indicates a possible inconsistency between how you generate the heatmaps and how you extract probabilities, or an error in the CSV labels.")

if __name__ == "__main__":
    main()