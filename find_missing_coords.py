import os
import pandas as pd

CONFIG = {
    "COORDS_DIR": "dataset/coords_metadata/",
    "TRAIN_CSV_PATH": "dataset/train.csv",
}

def find_missing_positive_samples():
    try:
        train_df = pd.read_csv(CONFIG["TRAIN_CSV_PATH"])
        positive_series_df = train_df[train_df['Aneurysm Present'] == 1]
        expected_positive_uids = set(positive_series_df['SeriesInstanceUID'])
        
        print(f"Total positive series that should exist: {len(expected_positive_uids)}")

        if not os.path.exists(CONFIG["COORDS_DIR"]):
            print(f"\nERROR: The coordinates directory does not exist at path:")
            print(f"   {CONFIG['COORDS_DIR']}")
            return

        generated_files = os.listdir(CONFIG["COORDS_DIR"])
        generated_uids = {f.replace('_coords.pkl', '') for f in generated_files if f.endswith('_coords.pkl')}
        
        print(f"Total coordinate files found: {len(generated_uids)}")

        missing_uids = expected_positive_uids - generated_uids
        
        print("-" * 50)
        
        if not missing_uids:
            print("\nSUCCESS! All positive samples have been correctly processed.")
        else:
            print(f"\nWARNING! {len(missing_uids)} positive samples were found that were not processed:")
            for uid in sorted(list(missing_uids)):
                print(f"  - {uid}")
            print("\nCheck these series to see why the preprocessing script failed.")

    except FileNotFoundError:
        print(f"\nERROR: The input file was not found at path:")
        print(f"   {CONFIG['TRAIN_CSV_PATH']}")
    except Exception as e:
        print(f"\nAn unexpected error has occurred: {e}")

if __name__ == "__main__":
    find_missing_positive_samples()