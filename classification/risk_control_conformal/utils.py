import sys
import os
import cv2
import torch

# Resolve paths (classification/ as project root for this part)
this_dir = os.path.dirname(os.path.abspath(__file__))                 # .../classification/risk_control_conformal
classification_dir = os.path.dirname(this_dir)                        # .../classification
upstream_dir = os.path.join(classification_dir, "upstream_xray_classifier")

if upstream_dir not in sys.path:
    sys.path.insert(0, upstream_dir)

import config 
from datasets import XRaysTestDataset


def load_trained_model(ckpt_name: str, device: torch.device):
    ckpt_path = os.path.join(upstream_dir, config.models_dir, ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model = checkpoint["model"]
    model.to(device)
    model.eval()
    return model

def get_val_df_and_classes(data_dir: str):
    """
    Use XRaysTestDataset, which is based on test_list.txt.
    test_list is your 'validation' pool now.

    Returns:
        class_names: list of all disease classes (in training order)
        val_df:      DataFrame of images and labels from test_list.txt
    """
    ds = XRaysTestDataset(data_dir, transform=config.transform)
    class_names = list(ds.all_classes)
    val_df = ds.test_df.reset_index(drop=True)
    return class_names, val_df

def preprocess_image_from_path(img_path: str):
    fname = os.path.basename(img_path)

    # always look for it in upstream_xray_classifier/data/images/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # classification/
    real_path = os.path.join(base_dir, "upstream_xray_classifier", "data", "images", fname)

    img_bgr = cv2.imread(real_path)
    if img_bgr is None:
        raise ValueError(f"Unable to read the image: {real_path}")

    return config.transform(img_bgr)