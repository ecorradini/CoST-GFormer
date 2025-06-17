# CoST-GFormer

CoST-GFormer is a complete prototype of a **Co**-evolving **S**patio-**T**emporal
**G**raph **Former** model.  It includes fully working modules for
spatio–temporal embeddings, attention mechanisms and memory buffers together
with a simple yet functional data pipeline.  While lightweight, the code can be
used end to end for experiments on real datasets.

## Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

The project relies on `numpy`, `torch`, `scipy` and a few helper libraries used
in the tests and GTFS loader.  `scipy` is required for the optional sparse
eigen decomposition used when computing spectral node embeddings.

## Running the demo

A short example using synthetic data is provided in `main.py`:

```bash
python main.py
```

This generates a tiny random dataset, builds the model and prints a few pieces
of information about the embeddings and predictions.

## Training on GTFS data

You can train the model on a GTFS (General Transit Feed Specification)
dataset using the script in `cost_gformer/train_gtfs.py`:

```bash
python -m cost_gformer.train_gtfs PATH_TO_STATIC_FEED [PATH_TO_REALTIME_FEED] [PATH_TO_VEHICLE_FEED]
```

Snapshot times throughout the library are expressed in seconds past midnight.
For example `08:00` would be represented as `8 * 3600`.

Crowding prediction requires that edge occupancy values are available. These
can be obtained from vehicle position feeds or supplied through a custom data
loader.

Convenience wrapper scripts are available so you don't have to type the
full Python command each time.  On Linux or macOS run

```bash
./train_gtfs.sh PATH_TO_STATIC_FEED [PATH_TO_REALTIME_FEED] [PATH_TO_VEHICLE_FEED] [options]
```

On Windows use

```
train_gtfs.bat PATH_TO_STATIC_FEED [PATH_TO_REALTIME_FEED] [PATH_TO_VEHICLE_FEED] [options]
```

Key command line options include:

- `--history` / `--horizon` – size of the input and forecast windows.
- `--epochs` – number of training epochs.
- `--lr` – learning rate for the SGD trainer.
- `--device` – choose `cpu` or `cuda`.
- `--regression` – use regression instead of classification for crowd level.

The spatio-temporal embedding module also exposes a `use_sparse` flag. When
enabled (the default) and the graph is large, spectral coordinates are computed
with `scipy.sparse.linalg.eigsh` for efficiency. Setting the flag to
``False`` forces a dense eigen decomposition instead.

The training script employs a lightweight NumPy based trainer.  When run on the
small GTFS example used in the unit tests the model reaches around 50&nbsp;s mean
absolute error for travel time and nearly perfect crowding classification
accuracy after a couple of epochs.

## Tests

Unit tests are located in the `tests/` directory and can be run with

```bash
pytest
```


## Sample GTFS dataset

A tiny GTFS feed used in the unit tests is provided under `sample_data/gtfs/`.
It contains just three stops and a single trip with real-time delays.  The
files can be used for quick experiments or to verify the evaluation script.

## Evaluation

Once a model has been trained and saved with `torch.save`, it can be evaluated
on a held-out GTFS dataset using `evaluate.py`:

```bash
python evaluate.py MODEL_PATH PATH_TO_STATIC_FEED [PATH_TO_REALTIME_FEED]
```

For example, after training on the sample data you might run:

```bash
python -m cost_gformer.train_gtfs sample_data/gtfs sample_data/gtfs/rt.pb --epochs 2
python - <<'PY'
import torch
from cost_gformer.trainer import Trainer
from cost_gformer.data import DataModule
from cost_gformer.gtfs import load_gtfs
from cost_gformer.model import CoSTGFormer

dataset = load_gtfs('sample_data/gtfs', 'sample_data/gtfs/rt.pb')
data = DataModule(dataset, history=3, horizon=1)
model = CoSTGFormer()
trainer = Trainer(model=model, data=data, epochs=2)
trainer.fit()
torch.save(trainer.model, 'model.pth')
PY
python evaluate.py model.pth sample_data/gtfs sample_data/gtfs/rt.pb
```

A well-trained model on this toy dataset should obtain around `50s` mean
absolute error, about `60s` RMSE and close to `100%` crowding accuracy.
Slight variations are expected depending on the random seed and training
settings.

