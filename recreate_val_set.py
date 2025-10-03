import pandas as pd
from sklearn.model_selection import GroupKFold

TRAIN_CSV_PATH = "dataset/preprocessed_3d_64_384_384/metadata_3d_384.csv"
N_SPLITS = 5 

TARGET_FOLD = 0
OUTPUT_CSV_PATH = "dataset/preprocessed_3d_64_384_384/replicated_holdout_fold0.csv"

print("--- Recreating the original validation set ---")
print(f"Loading data from: {TRAIN_CSV_PATH}")

try:
    df = pd.read_csv(TRAIN_CSV_PATH)
except FileNotFoundError:
    print(f"ERROR: File not found at '{TRAIN_CSV_PATH}'. Please check the path.")
    exit()

gkf = GroupKFold(n_splits=N_SPLITS)

splits = gkf.split(df, groups=df['SeriesInstanceUID'])

try:
    _train_idx, val_idx = next(s for i, s in enumerate(splits) if i == TARGET_FOLD)
except StopIteration:
    print(f"ERROR: Fold {TARGET_FOLD} is not valid for n_splits={N_SPLITS}. Valid folds range from 0 to {N_SPLITS-1}.")
    exit()

val_df_original = df.iloc[val_idx]
print(f"Found {len(val_df_original)} rows corresponding to {val_df_original['SeriesInstanceUID'].nunique()} unique series in the validation set for fold {TARGET_FOLD}.")

val_df_original.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"Holdout recreated with all columns and saved to: {OUTPUT_CSV_PATH}")