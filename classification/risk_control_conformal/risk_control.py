"""LTT FDR experiment (using test_list as validation)"""

import os
import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

#from datasets import XRaysTestDataset
from utils import load_trained_model, get_val_df_and_classes, preprocess_image_from_path

# base dir = classification/
this_dir = os.path.dirname(os.path.abspath(__file__))   # .../classification/risk_control_conformal
base_dir = os.path.dirname(this_dir)                    # .../classification
upstream_dir = os.path.join(base_dir, "upstream_xray_classifier")

def run_fdr_experiment(
    ckpt_name: str,
    data_path: str = ".",
    alpha: float = 0.1,
    delta: float = 0.1,
    num_lambdas: int = 101,
    calib_frac: float = 0.5,
    seed: int = 0,
    num_splits: int = 1,
):
    """
    If num_splits == 1:
        run a single LTT+FDR experiment (as before).
    If num_splits > 1:
        perform 'coverage check' on the LAST split,
        save images that have >1 predicted class (sigmoid > lambda_star)
        into results/predicted_images.
    """

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    # dataset root = classification/upstream_xray_classifier/data/<data_path>
    data_dir = os.path.join(upstream_dir, "data", data_path)

    # Validation pool: test_list.txt to XRaysTestDataset.test_df
    class_names, val_pool_df = get_val_df_and_classes(data_dir)
    num_classes = len(class_names)
    print(f"Using {len(val_pool_df)} validation examples from test_list.txt")
    print(f"Number of classes: {num_classes}")

    # label -> index
    label_to_idx = {lab: i for i, lab in enumerate(class_names)}

    # Multi-hot labels Y (cache)
    n_total = len(val_pool_df)
    Y = torch.zeros((n_total, num_classes), dtype=torch.bool)
    for i, row in val_pool_df.iterrows():
        labels = str(row["Finding Labels"]).split("|")
        for lab in labels:
            if lab in label_to_idx:
                Y[i, label_to_idx[lab]] = True

    # Load model and PRECOMPUTE SIGMOIDS for every example (big cache)
    print("\nPrecomputing model probabilities (sigmoids) for all validation images ...")
    model = load_trained_model(ckpt_name, device)
    model.eval()
    sig_cache = torch.zeros((n_total, num_classes), device=device)

    with torch.no_grad():
        for i, row in tqdm(list(enumerate(val_pool_df.itertuples())), total=n_total):
            img_path = row.image_links
            img_tensor = preprocess_image_from_path(img_path).unsqueeze(0).to(device)
            sig = model(img_tensor).sigmoid().squeeze(0) 
            sig_cache[i] = sig

    # Common lambda grid
    lambdas = torch.linspace(0.0, 1.0, num_lambdas, device=device)

    # For FDR check
    rng = np.random.default_rng(seed)
    fdr_over_splits = []
    lambda_star_list  = []

    # dir for chosen images (for last split)
    chosen_dir = os.path.join(base_dir, "risk_control_conformal", "results", "predicted_images")
    print(f'######################################################################################### {chosen_dir}')
    os.makedirs(chosen_dir, exist_ok=True)

    # ==== MAIN LOOP OVER RANDOM SPLITS ====
    for r in range(num_splits):
        print(f"\n========== Split {r+1}/{num_splits} ==========")

        # --- random split into calibration + evaluation ---
        perm = rng.permutation(n_total)
        n_calib = int(calib_frac * n_total)
        calib_idx = perm[:n_calib]
        eval_idx = perm[n_calib:]

        print(f"Calibration size: {len(calib_idx)}, evaluation size: {len(eval_idx)}")

        # ----- LTT on calibration (Figure 19 style, using cached sigmoids) -----
        losses = torch.zeros((len(calib_idx), num_lambdas), device=device)

        print("Computing calibration losses (FDP) over λ-grid ...")
        with torch.no_grad():
            for row_i, idx in tqdm(list(enumerate(calib_idx)), total=len(calib_idx)):
                sigmoids = sig_cache[idx]         # [K], from cache
                y_mask   = Y[idx].to(device)      # [K]

                for j in range(num_lambdas):
                    T = sigmoids > lambdas[j]
                    set_size = T.float().sum()
                    if set_size != 0:
                        tp = torch.logical_and(T, y_mask).float().sum()
                        losses[row_i, j] = 1.0 - tp / set_size  # FDP loss

        risk = losses.mean(dim=0)  # \hat R(λ)

        n_cal = len(calib_idx)
        diff = torch.relu(alpha - risk)
        pvals = torch.exp(-2.0 * n_cal * diff * diff)

        # Bonferroni over λ-grid (still conservative)
        below_delta = (pvals <= delta).float()  # / num_lambdas
        valid = torch.tensor(
            [(below_delta[j:].mean().item() == 1.0) for j in range(num_lambdas)],
            dtype=torch.bool,
            device=device,
        )
        lambda_set = lambdas[valid]

        if lambda_set.numel() == 0:
            print("No λ passed the FDR test at this split – skipping split.")
            continue

        lambda_star = lambda_set.min().item()
        lambda_star_list.append(lambda_star)
        print(f"Number of valid λ's: {lambda_set.numel()}")
        print(f"Chosen λ* (min valid λ) = {lambda_star:.4f}")

        # ----- Evaluate empirical FDR on evaluation subset (using cache) -----
        fdp_list = []
        with torch.no_grad():
            for idx in tqdm(eval_idx):
                sigmoids = sig_cache[idx]
                y_mask   = Y[idx].to(device)
                T = sigmoids > lambda_star
                set_size = T.float().sum()

                # For the final split, save images with >1 predicted class ===
                if (r == num_splits - 1) and (set_size > 1):
                    # predicted class indices
                    pred_idx = torch.where(T)[0].tolist()
                    pred_labels = [class_names[k] for k in pred_idx]
                    # build filename: labels joined by "_", spaces removed
                    label_str = "_".join(lbl.replace(" ", "") for lbl in pred_labels)

                    orig_path = val_pool_df.iloc[idx]["image_links"]
                    img_bgr = cv2.imread(orig_path)

                    base_name = os.path.basename(orig_path)
                    out_name = f"{label_str}_{base_name}"
                    out_path = os.path.join(chosen_dir, out_name)
                    cv2.imwrite(out_path, img_bgr)

                # FDP computation
                if set_size != 0:
                    tp = torch.logical_and(T, y_mask).float().sum()
                    fdp = 1.0 - tp / set_size
                else:
                    fdp = torch.tensor(0.0, device=device)
                fdp_list.append(fdp)

        fdr_est = torch.stack(fdp_list).mean().item()
        print(f"Empirical FDR on evaluation (split {r+1}) ≈ {fdr_est:.4f}")
        fdr_over_splits.append(fdr_est)

    # After all splits 
    if len(fdr_over_splits) == 0:
        print("\nNo successful splits (no λ passed in every run). Nothing to plot.")
        return

    fdr_over_splits = np.array(fdr_over_splits)
    print("\n===== Coverage check over splits =====")
    print(f"Number of successful splits: {len(fdr_over_splits)}")
    print(f"median λ: {np.median(np.array(lambda_star_list)):.4f}")
    print(f"Mean empirical FDR: {fdr_over_splits.mean():.4f}")
    print(f"Std  empirical FDR: {fdr_over_splits.std(ddof=1):.4f}")
    frac_viol = np.mean(fdr_over_splits > alpha)
    print(f"Fraction of splits with FDR > alpha ({alpha}): {frac_viol:.3f} "
          f"(should be =< δ = {delta})")

    # Histogram (like the paper, but for FDR)
    plt.figure(figsize=(6, 4))
    plt.hist(fdr_over_splits, bins=15)
    plt.axvline(alpha, linestyle="--", linewidth=2, label=f"α = {alpha:.2f}")
    plt.axvline(fdr_over_splits.mean(), linestyle=":", linewidth=2, label=f"mean = {fdr_over_splits.mean():.2f}")
    plt.xlabel("Empirical FDR on evaluation")
    plt.ylabel("Count over splits")
    plt.title(f"Distribution of FDR over {len(fdr_over_splits)} splits")
    plt.legend()
    plt.tight_layout()
    out_path = chosen_dir = os.path.join(base_dir, "risk_control_conformal", "results", "fdr_coverage_hist.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved FDR coverage histogram to {out_path}")
    print(f"Chosen images (last split, |T|>1) saved to: {chosen_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="NIH Chest X-ray multi-label LTT FDR control experiments\n"
    )
    parser.add_argument("--num_splits", 
                        type=int, 
                        default=1,
                        help="Number of random calib/eval splits for coverage check")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stage1_0001_03.pth",
        help="Checkpoint filename in models/ directory",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=".",
        help="Subdirectory inside ./data containing NIH dataset (default: '.')",
    )
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--num_lambdas", type=int, default=101)
    parser.add_argument("--calib_frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    run_fdr_experiment(
        ckpt_name=args.ckpt,
        data_path=args.data_path,
        num_splits = args.num_splits,
        alpha=args.alpha,
        delta=args.delta,
        num_lambdas=args.num_lambdas,
        calib_frac=args.calib_frac,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
