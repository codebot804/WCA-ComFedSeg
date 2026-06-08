# Data

Raw datasets are not redistributed in this repository. Download them from their official public sources and place them under `data/raw/` before running real-dataset experiments.

Expected local layout:

```text
data/raw/BUSI/Dataset_BUSI_with_GT/
  benign/
  malignant/
  normal/        # deferred in current experiments

data/raw/BUS-UCLM/
  ...            # use the public BUS-UCLM release; normal cases are deferred

data/raw/Kvasir-SEG/
  images/
  masks/
```

The current manuscript uses benign/malignant BUSI and BUS-UCLM cases only. Normal cases require separate empty-mask handling and are not part of the reported claims.
