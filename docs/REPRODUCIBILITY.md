# Reproducibility Notes

The repository uses command-line arguments and CSV/JSON logs for reproducibility.

## Main reported budget

- clients: 3
- rounds: 10
- local epochs: 2
- batch size: 4
- image size: 128
- seeds: 42, 123, 2025
- optimizer: Adam
- learning rate: 1e-3
- segmentation loss: BCE-Dice loss
- FedProx coefficient: 0.01
- WCA alpha: 0.5
- WCA-Comm client fraction: 0.67
- WCA-Comm minimum selected clients: 2

## Output locations

Each run writes:

- `metrics.csv`
- `summary.json`
- `args.json`
- `split_config.json`
- WCA weight logs for WCA variants
- communication logs for WCA-Comm variants

Processed paper tables and figures are stored under `paper/source_data/`, `paper/tables/`, and `paper/figures/`.
