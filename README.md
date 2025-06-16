# CoST-GFormer

CoST-GFormer is a minimal demonstration of a **Co**-evolving **S**patio-**T**emporal
**G**raph **Former** model. The code provides lightweight components for
spatio–temporal embeddings, attention modules and memory units alongside a
simple data pipeline.  It is primarily intended for small examples and unit
tests and does not aim to be a full production implementation.

## Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

The requirements are purposely minimal and rely only on `numpy`, `torch` and a
few helper libraries used in the tests and GTFS loader.

## Running the demo

A short example using synthetic data is provided in `main.py`:

```bash
python main.py
```

This generates a tiny random dataset, builds the model and prints a few pieces
of information about the embeddings and predictions.

## Training on GTFS data

For a more realistic experiment you can train the model on a GTFS (General
Transit Feed Specification) dataset using the script in
`cost_gformer/train_gtfs.py`:

```bash
python -m cost_gformer.train_gtfs PATH_TO_STATIC_FEED [PATH_TO_REALTIME_FEED]
```

Key command line options include:

- `--history` / `--horizon` – size of the input and forecast windows.
- `--epochs` – number of training epochs.
- `--lr` – learning rate for the tiny SGD trainer.
- `--device` – choose `cpu` or `cuda`.
- `--regression` – use regression instead of classification for crowd level.

The script uses a very small trainer written with NumPy and is intended for
experiments or unit tests rather than large scale training.

## Tests

Unit tests are located in the `tests/` directory and can be run with

```bash
pytest
```

