# WCA-ComFedSeg

Worst-Client-Aware Communication-Efficient Federated Learning for Non-IID Breast Ultrasound Image Segmentation.

This repository contains a lightweight PyTorch implementation for simulated federated binary medical image segmentation. The main empirical setting evaluates whether a worst-client-aware aggregation and communication scheduler can improve client-level fairness while reducing uploaded model-update traffic.

## Scope

The current evidence supports a scoped empirical finding: on BUSI benign/malignant moderate non-IID splits, WCA-Comm improves average Dice, worst-client Dice, and best-worst client gap over FedAvg, FedProx, and FedBN while reducing uploaded communication by 30%. BUS-UCLM, BUSI hard non-IID, and Kvasir-SEG are included as boundary or diagnostic settings rather than broad success claims.

## Repository Structure

```text
configs/      Experiment and method registry YAML files
datasets/     Dataset discovery, preprocessing, and client-split loaders
federated/    Federated client/server, aggregation, FedBN, and communication logic
methods/      Local and centralized training utilities
models/       Lightweight 2D U-Net backbone
scripts/      Validation, comparison, diagnosis, and paper-preparation utilities
tests/        Unit tests for metrics, scheduler, and WCA components
utils/        Losses, metrics, logging, config, and visualization helpers
results/      Selected processed summaries used by manuscript tables/figures
paper/        Figure/table source data, final figures, and manuscript materials
data/         Local data placeholder; raw datasets are not redistributed
```

## Environment

The reported experiments were run in a CPU-only Python environment. The implementation uses PyTorch and standard scientific Python packages.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

The experiments reported in the manuscript used Python 3.12.7 and PyTorch 2.12.0+cpu. Other PyTorch versions may work, but exact reproducibility should use the recorded environment.

## Data

Raw medical datasets are not included. Download public datasets from their original sources and place them under `data/raw/`. See `data/README.md` for the expected layout.

Used/evaluated datasets:

- BUSI breast ultrasound dataset: benign and malignant cases used; normal cases deferred.
- BUS-UCLM breast ultrasound dataset: benign and malignant cases used for boundary analysis; normal cases deferred.
- Kvasir-SEG: gastrointestinal polyp segmentation used as a cross-modality boundary setting.

## Example Commands

Synthetic smoke test:

```bash
python run_experiment.py --method fedavg --dataset synthetic --split moderate_noniid --clients 3 --rounds 3 --local-epochs 1 --batch-size 4 --image-size 128 --device cpu
```

BUSI moderate non-IID example with WCA-Comm:

```bash
python run_experiment.py \
  --method wca_comfedseg_comm \
  --dataset busi \
  --split moderate_noniid \
  --clients 3 \
  --rounds 10 \
  --local-epochs 2 \
  --batch-size 4 \
  --image-size 128 \
  --seed 42 \
  --device cpu \
  --results-dir results/logs/phase5m_moderate_multiseed
```

Baseline methods use the same runner by changing `--method` to `fedavg`, `fedprox`, or `fedbn`.

## Main Metrics

The evaluation reports:

- average Dice and IoU,
- worst-client Dice,
- client Dice standard deviation,
- best-worst client gap,
- uploaded model-update traffic in MB,
- communication reduction relative to full-participation baselines.

Dice and IoU are computed from sigmoid-thresholded binary predictions at threshold 0.5.

## Tests

```bash
pytest tests
```

## Notes on Claims

This codebase should not be interpreted as a universal medical segmentation method. The strongest current result is the BUSI moderate non-IID setting. Other datasets and stronger heterogeneity settings are included to identify method boundaries and failure modes.

## Citation

If you use this repository, please cite the accompanying manuscript or acknowledge the repository once a public release record is available.
