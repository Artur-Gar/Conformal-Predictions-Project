# Risk-Control Conformal for Classification (NIH Chest X-rays)

This module contains two parts:

- **`upstream_xray_classifier/`** — the baseline **multi-label chest X-ray classifier** (ResNet50, trained with Focal Loss).  
- **`risk_control_conformal/`** — **distribution-free risk control** (Learn-Then-Test / LTT) on top of the classifier, selecting a threshold $\lambda$ to control **False Discovery Rate (FDR)** for multi-label prediction sets.

## Quick start

Install dependencies (from `classification/`):
```bash
pip install -r requirements.txt
```

---

## Workflow

### 1) (Optional) Data split
If you need to generate `train_val_list.txt` / `test_list.txt`, run the data-prep script in `data_prep/`  
(see `data_prep/README.md` for details).

### 2) Train / test baseline model
Train and evaluate the ResNet baseline in `upstream_xray_classifier/`  
(see `upstream_xray_classifier/README.md` for the exact commands).

Checkpoints are saved to:
- `upstream_xray_classifier/models/`

### 3) Run conformal / risk control experiments
Run DFRC / LTT experiments in `risk_control_conformal/`  
(see `risk_control_conformal/README.md`).

Outputs are saved to:
- `risk_control_conformal/results/`
  - `fdr_coverage_hist.png` — histogram across random splits
  - `predicted_images/` — example multi-label predictions saved from the last split

---

## Where to look
- Baseline model details & training: `upstream_xray_classifier/README.md`
- Conformal / DFRC experiments & results: `risk_control_conformal/README.md`
"""
