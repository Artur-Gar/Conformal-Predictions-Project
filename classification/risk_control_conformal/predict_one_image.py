"""predict labels for one image """

import os
import sys
import cv2
import numpy as np
import torch
import argparse

from utils import load_trained_model, get_val_df_and_classes

# Resolve paths + make upstream_xray_classifier importable
this_dir = os.path.dirname(os.path.abspath(__file__))          # .../classification/risk_control_conformal
classification_dir = os.path.dirname(this_dir)                 # .../classification
upstream_dir = os.path.join(classification_dir, "upstream_xray_classifier")

if upstream_dir not in sys.path:
    sys.path.insert(0, upstream_dir)

import config


def predict_single_image(
    img_name: str,
    ckpt_name: str,
    data_path: str = ".",
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    data_dir = os.path.join(upstream_dir, "data", data_path)

    # classes come from XRaysTestDataset / disease_classes.pkl
    class_names, _ = get_val_df_and_classes(data_dir)
    num_classes = len(class_names)
    print(f"Detected {num_classes} categories: {class_names}")

    model = load_trained_model(ckpt_name, device)

    img_path = os.path.join(upstream_dir, "data", "xrays_to_predict", img_name)
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Unable to read the image: {img_path}")
    img_tensor = config.transform(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)          # [1, K]
        probs = torch.sigmoid(logits)[0]    # [K]
        probs = probs.cpu().numpy()

    print("\n===== Probability of various diseases (from highest to lowest) =====")
    sorted_idx = np.argsort(-probs)
    for idx in sorted_idx:
        cls_name = class_names[idx]
        p = probs[idx]
        print(f"{cls_name:20s} : {p:.3f} ({p*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="NIH Chest X-ray single-image multi-label prediction\n"
    )
    parser.add_argument(
        "--image_name",
        type=str,
        help="X-ray image filename",
    )
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

    args = parser.parse_args()

    if args.image_name is None:
            raise ValueError("--image_name is required")
    else:
        predict_single_image(
            img_name=args.image_name,
            ckpt_name=args.ckpt,
            data_path=args.data_path,
        )


if __name__ == "__main__":
    main()
