#!/usr/bin/env sh
# Wrapper script for running CoST-GFormer training on a GTFS dataset.
# Usage: ./train_gtfs.sh STATIC_FEED [REALTIME_FEED] [VEHICLE_FEED] [options]
# All arguments are passed directly to the Python entry point.

# Detect a Python interpreter
if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "Error: Python is not installed." >&2
    exit 1
fi

exec "$PYTHON" -m cost_gformer.train_gtfs "$@"
