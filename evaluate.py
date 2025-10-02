import argparse
import pandas as pd
from copy import copy
import importlib
import torch

from src.data.utils import get_dataset, get_dataloader
from src.models.utils import get_model
from src.modules.utils import run_eval

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model checkpoint.")
    parser.add_argument("--config", type=str, default="eval_config", help="Configuration file name")
    args = parser.parse_args()

    print("--- Starting model evaluation ---")
    
    cfg = copy(importlib.import_module(f'src.configs.{args.config}').cfg)
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.local_rank = 0
    print(f"Loaded config: {args.config}")
    print(f"Loading checkpoint: {cfg.weights_path}")

    eval_df = pd.read_csv(cfg.eval_csv_path)
    eval_ds = get_dataset(eval_df, cfg, mode="val")
    eval_dl = get_dataloader(eval_ds, cfg, mode="val")
    print(f"Loaded {len(eval_ds)} samples for evaluation.")

    model, _ = get_model(cfg, inference_mode=True)
    model.to(cfg.device)

    metrics = run_eval(model, eval_df, eval_dl, cfg)

    val_metrics = metrics['val']
    score = val_metrics['score']
    loss = val_metrics['loss']
    auc_present = val_metrics['auc_present']
    mean_location_auc = val_metrics['mean_location_auc']
    
    print("\n--- Evaluation results ---")
    print(f"  Overall score (weighted AUC): {score:.4f}")
    print(f"  Mean validation loss:         {loss:.4f}")
    print(f"  'Aneurysm present' AUC:       {auc_present:.4f}")
    print(f"  Mean location AUC:            {mean_location_auc:.4f}")
    print("--------------------------\n")

if __name__ == "__main__":
    main()