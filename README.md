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

The project relies only on `numpy`, `torch` and a few helper libraries used in
the tests and GTFS loader.

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
python -m cost_gformer.train_gtfs PATH_TO_STATIC_FEED [PATH_TO_REALTIME_FEED]
```

Key command line options include:

- `--history` / `--horizon` – size of the input and forecast windows.
- `--epochs` – number of training epochs.
- `--lr` – learning rate for the SGD trainer.
- `--device` – choose `cpu` or `cuda`.
- `--regression` – use regression instead of classification for crowd level.

The training script employs a lightweight NumPy based trainer.  When run on the
small GTFS example used in the unit tests the model reaches around 50&nbsp;s mean
absolute error for travel time and nearly perfect crowding classification
accuracy after a couple of epochs.

## Tests

Unit tests are located in the `tests/` directory and can be run with

```bash
pytest
```

