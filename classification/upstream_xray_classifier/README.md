# NIH Chest X-ray multi-label baseline (ResNet-50)

This classification baseline is adapted from the original GitHub repository:
- https://github.com/n0obcoder/NIH-Chest-X-Rays-Multi-Label-Image-Classification-In-Pytorch

We use this baseline as the **upstream classifier** for our conformal prediction experiments (risk control / FDR control on multi-label prediction sets).

---

## Dataset

We use the **NIH Chest X-ray** dataset (Data_Entry_2017.csv):
- https://www.kaggle.com/nih-chest-xrays/data#Data_Entry_2017.csv

**Labels (15 total):** 14 diseases + **No Finding**.  
Possible labels include:
- Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pneumonia, Pleural_thickening, Cardiomegaly, Nodule, Mass, Hernia, and No Finding.

### Subsample used in this project
To keep experiments manageable, we train on a subset:
- **5606 images** from the Kaggle sample dataset: https://www.kaggle.com/datasets/nih-chest-xrays/sample
- plus **1944 additional images** from the original dataset

**Total:** 7550 X-ray images (1024×1024).

---

## Model

- **Backbone:** pretrained **ResNet-50**
- **Head:** final fully-connected layer replaced with a 15-dimensional output (multi-label logits)
- **Activation:** sigmoid per class (multi-label)

---

## Loss

- **Focal Loss** (used throughout)

---

## Training

We train in stages (transfer learning), progressively fine-tuning the network.

### Stage 1 (train from scratch)
- **Trainable layers:** `layer2`, `layer3`, `layer4`, `fc`
- **Loss:** FocalLoss
- **Learning rate:** 1e-4
- **Batch size:** 32
- **Epochs:** 4

Command:
```bash
python main.py --bs 32 --lr 1e-4
```

### Stage 2 (resume from checkpoint)
- **Resume from:** stage1_0001_04.pth
- **Stage argument:** --stage 2
- **Loss:** FocalLoss
- **Learning rate:** 1e-4
- **Batch size:** 32
- **Epochs:** 4

Command:
```bash
python main.py --resume --ckpt stage1_0001_04.pth --stage 2 --bs 32 --lr 1e-4
```

---

## Testing

To run evaluation using a saved checkpoint:

```bash
python main.py --test --ckpt stage1_0001_08.pth
```

---

## Result
On our subset, the baseline achieves an average **ROC AUC ≈ 0.739** across disease classes (excluding No Finding).
