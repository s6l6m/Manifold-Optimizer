# Manifold Optimizer Experiment

## prerequisit
- NVIDIA GPU with CUDA 13 installed

## Setup enviroment

**Option A: uv**

```bash
uv sync
```

**Option B: plain venv (python 3.11)**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Install RAPIDS cuML
follow the guide here [rapids cuml](https://docs.rapids.ai/install/)

## Run notebooks inside the enviroment

**Jupyter server**:

```bash
jupyter notebook .
```

**VSCode**: select the `.venv` or uv interpreter as the kernel in the kernel picker.

## Configuration

Each notebook imports `CONFIG` from [config.py](config.py). Override parameters by passing keyword arguments:

```python
from config import Config

CONFIG = Config(
    DB_NAME="my_db",
)
```

## Data

Place CSV files inside [data/](data/):

```
data/
  120_subclasses_chemical_subset.csv
```

The path is resolved via `CONFIG.DATA_DIR_PATH`, so if your file has a different name please update the `pd.read_csv(...)` call in [insertion.ipynb](insertion.ipynb).

**Please run at least one insertion block from the insertion notebook into a parquet file and adjust the config to run the experiment!!**