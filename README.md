# Conformal Predictions Research

This repository contains experiments with **conformal prediction** methods for two settings:

- **Classification** (multi-label image classification + risk control / DFRC)
- **Regression** (time-series / forecasting setting)

## Repository structure
- `classification/` — NIH Chest X-ray multi-label classifier + conformal/risk-control experiments  
  See `classification/README.md` for how to train, evaluate, and where results are saved.

- `regression/` — conformal prediction for regression (time-series) experiments  
  See `regression/README.md` for setup, runs, and outputs.

Each folder is self-contained and includes its own README with commands and results.

## Report
[research_report.pdf](research_report.pdf) - outlines the theoretical basis and presents a comprehensive overview of the research and the results.
