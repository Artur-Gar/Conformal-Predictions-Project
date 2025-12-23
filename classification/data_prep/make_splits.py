"""
make_splits.py — generate train/test lists for NIH Chest X-ray (multi-label).

This repo's structure (default):
classification/
  upstream_xray_classifier/
    data/
        Data_Entry_2017.csv
        images/
        *.png / *.jpg ...
        (outputs will be saved here)
  data_prep/
    make_splits.py

What this script does:
1) Loads Data_Entry_2017.csv and keeps only rows whose image file exists in data/images/.
2) Finds the largest frequency threshold THRESH such that dropping label-combinations
   with count < THRESH removes <= max_drop_pct% of samples.
3) Saves dropped (rare) rows to rare_samples.csv.
4) Stratified 80/20 split on remaining rows (stratify by full "Finding Labels" string).
5) Saves train_val_list.txt and test_list.txt into upstream_xray_classifier/data/.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():

    this_dir = os.path.dirname(os.path.abspath(__file__))
    classification_dir = os.path.dirname(this_dir) 
    
    data_dir = os.path.join(classification_dir, "upstream_xray_classifier", "data")
    csv_path = os.path.join(data_dir, "Data_Entry_2017.csv")
    img_dir = os.path.join(data_dir, "images")

    # ------------------------------------------------------------------
    # 1. Load CSV and keep only rows with existing images
    # ------------------------------------------------------------------
    df = pd.read_csv(csv_path)

    have_imgs = set(os.listdir(img_dir))
    df = df[df["Image Index"].isin(have_imgs)].copy()

    y = df["Finding Labels"]
    vc = y.value_counts()

    total = len(df)
    print(f"Total rows with images: {total}")

    # ------------------------------------------------------------------
    # 2. Search for the largest THRESH that drops <= 5% of the data
    #    rare_combos = label combinations with frequency < THRESH
    # ------------------------------------------------------------------
    best_thresh = None
    best_drop_pct = None

    print("Trying thresholds:")
    for th in range(2, 21):   # thresholds 2..20, adjust range if you want
        rare_combos = vc[vc < th].index
        num_rare_rows = df[df["Finding Labels"].isin(rare_combos)].shape[0]
        drop_pct = 100 * num_rare_rows / total
        print(f"  THRESH={th}: drop {num_rare_rows} rows ({drop_pct:.2f}%)")

        if drop_pct <= 5.0:
            best_thresh = th
            best_drop_pct = drop_pct
        else:
            print("  Exceeds 5%")
            break

    # If nothing satisfies <= 5%, fall back to 2 and warn
    if best_thresh is None:
        best_thresh = 2
        rare_combos = vc[vc < best_thresh].index
        num_rare_rows = df[df["Finding Labels"].isin(rare_combos)].shape[0]
        best_drop_pct = 100 * num_rare_rows / total
        print("\nNo threshold between 2 and 20 keeps drop <= 5%.")
        print("Falling back to THRESH=2.")
    else:
        print(f"\nChosen THRESH = {best_thresh} (drops {best_drop_pct:.2f}% of data)")

    # ------------------------------------------------------------------
    # 3. Using chosen THRESH: identify rare combos and masks
    # ------------------------------------------------------------------
    rare_combos = vc[vc < best_thresh].index

    rare_mask = df["Finding Labels"].isin(rare_combos)
    rare_df = df[rare_mask].copy()
    rare_path = os.path.join(data_dir, "rare_samples.csv")
    rare_df.to_csv(rare_path, index=False)
    print(f"Saved {len(rare_df)} rare samples to {rare_path}")

    # Non-rare (kept) data
    df_clean = df[~rare_mask].copy()
    y_clean = df_clean["Finding Labels"]
    print(f"Remaining (non-rare) samples: {len(df_clean)}")

    # ------------------------------------------------------------------
    # 4. Stratified train/test split on non-rare data
    # ------------------------------------------------------------------
    train_idx, test_idx = train_test_split(
        df_clean["Image Index"],
        test_size=0.2,
        random_state=42,
        stratify=y_clean
    )

    # ------------------------------------------------------------------
    # 5. Save lists for your PyTorch dataset code
    # ------------------------------------------------------------------
    train_list_path = os.path.join(data_dir, "train_val_list.txt")
    test_list_path = os.path.join(data_dir, "test_list.txt")

    with open(train_list_path, "w") as f:
        f.write("\n".join(train_idx))

    with open(test_list_path, "w") as f:
        f.write("\n".join(test_idx))

    print(f"Train size: {len(train_idx)},  Test size: {len(test_idx)}")
    print(f"Saved train list to {train_list_path}")
    print(f"Saved test list to {test_list_path}")

if __name__ == "__main__":
    main()
