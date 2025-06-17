from __future__ import annotations

"""Utilities for loading GTFS feeds into :class:`GraphSnapshot` sequences."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
import csv
import io
import os
import zipfile

import numpy as np
from google.transit import gtfs_realtime_pb2

from .data import GraphSnapshot, DynamicGraphDataset


# ---------------------------------------------------------------------------
# Helper parsing utilities
# ---------------------------------------------------------------------------

def _open_gtfs_file(path: str, name: str):
    """Return a text file handle for a file within a GTFS feed."""
    if os.path.isdir(path):
        return open(os.path.join(path, name), "r", encoding="utf-8")
    if zipfile.is_zipfile(path):
        zf = zipfile.ZipFile(path)
        return io.TextIOWrapper(zf.open(name), encoding="utf-8")
    raise FileNotFoundError(path)


def _parse_time(value: str) -> int:
    h, m, s = value.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


@dataclass
class Segment:
    trip_id: str
    stop_seq: int
    u: int
    v: int
    depart: int
    arrive: int


def parse_gtfs_static(path: str) -> Tuple[Dict[str, int], List[Segment]]:
    """Parse a minimal subset of a GTFS static feed."""

    with _open_gtfs_file(path, "stops.txt") as f:
        reader = csv.DictReader(f)
        stop_map = {row["stop_id"]: idx for idx, row in enumerate(reader)}

    stop_times: Dict[str, List[Tuple[int, str, int, int]]] = {}
    with _open_gtfs_file(path, "stop_times.txt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trip = row["trip_id"]
            seq = int(row["stop_sequence"])
            stop_id = row["stop_id"]
            arr = _parse_time(row["arrival_time"])
            dep = _parse_time(row["departure_time"])
            stop_times.setdefault(trip, []).append((seq, stop_id, arr, dep))

    for lst in stop_times.values():
        lst.sort(key=lambda x: x[0])

    segments: List[Segment] = []
    for trip_id, lst in stop_times.items():
        for i in range(len(lst) - 1):
            seq, sid, _, dep = lst[i]
            _, sid_next, arr_next, _ = lst[i + 1]
            u = stop_map[sid]
            v = stop_map[sid_next]
            segments.append(Segment(trip_id, seq, u, v, dep, arr_next))

    return stop_map, segments


def parse_gtfs_realtime(path: str) -> Dict[Tuple[str, int], int]:
    """Parse GTFS real-time TripUpdate delays."""

    feed = gtfs_realtime_pb2.FeedMessage()
    with open(path, "rb") as f:
        feed.ParseFromString(f.read())

    delays: Dict[Tuple[str, int], int] = {}
    for ent in feed.entity:
        if not ent.HasField("trip_update"):
            continue
        tu = ent.trip_update
        trip_id = tu.trip.trip_id
        for stu in tu.stop_time_update:
            seq = stu.stop_sequence
            delay = 0
            if stu.HasField("departure") and stu.departure.HasField("delay"):
                delay = stu.departure.delay
            elif stu.HasField("arrival") and stu.arrival.HasField("delay"):
                delay = stu.arrival.delay
            delays[(trip_id, seq)] = delay
    return delays


def parse_vehicle_positions(path: str) -> Dict[Tuple[str, int], float]:
    """Parse GTFS real-time VehiclePosition occupancies."""

    feed = gtfs_realtime_pb2.FeedMessage()
    with open(path, "rb") as f:
        feed.ParseFromString(f.read())

    occupancies: Dict[Tuple[str, int], float] = {}
    for ent in feed.entity:
        if not ent.HasField("vehicle"):
            continue
        vp = ent.vehicle
        trip_id = vp.trip.trip_id
        if not trip_id:
            continue

        seq = 0
        if vp.current_stop_sequence > 0:
            seq = vp.current_stop_sequence
        elif vp.stop_id:
            seq = 0  # unknown sequence
        if seq <= 0:
            continue

        if vp.HasField("occupancy_percentage"):
            value = float(vp.occupancy_percentage)
        elif vp.HasField("occupancy_status"):
            value = float(int(vp.occupancy_status))
        else:
            value = 0.0

        occupancies[(trip_id, seq - 1)] = value

    return occupancies


# ---------------------------------------------------------------------------
# Main conversion function
# ---------------------------------------------------------------------------

def build_snapshots(
    segments: Iterable[Segment],
    delays: Optional[Dict[Tuple[str, int], int]] = None,
    occupancies: Optional[Dict[Tuple[str, int], float]] = None,
) -> DynamicGraphDataset:
    """Convert segments to :class:`DynamicGraphDataset`."""

    delays = delays or {}
    occupancies = occupancies or {}

    by_time: Dict[int, List[Segment]] = {}
    for seg in segments:
        by_time.setdefault(seg.depart, []).append(seg)

    snapshots: List[GraphSnapshot] = []
    for t in sorted(by_time.keys()):
        edges: List[Tuple[int, int]] = []
        static_feat: Dict[Tuple[int, int], np.ndarray] = {}
        dyn_feat: Dict[Tuple[int, int], np.ndarray] = {}
        for seg in by_time[t]:
            e = (seg.u, seg.v)
            edges.append(e)
            travel = seg.arrive - seg.depart
            static_feat[e] = np.array([float(travel)], dtype=np.float32)
            delay = float(delays.get((seg.trip_id, seg.stop_seq), 0))
            occ = float(occupancies.get((seg.trip_id, seg.stop_seq), 0.0))
            dyn_feat[e] = np.array([delay, occ], dtype=np.float32)
        snap = GraphSnapshot(
            time=t,
            edges=edges,
            static_edge_feat=static_feat,
            dynamic_edge_feat=dyn_feat,
        )
        snapshots.append(snap)

    return DynamicGraphDataset(snapshots)


def load_gtfs(
    static_path: str,
    realtime_path: Optional[str] = None,
    vehicle_path: Optional[str] = None,
) -> DynamicGraphDataset:
    """High level loader for GTFS feeds."""

    _, segments = parse_gtfs_static(static_path)
    delays = parse_gtfs_realtime(realtime_path) if realtime_path else None
    occ = parse_vehicle_positions(vehicle_path) if vehicle_path else None
    return build_snapshots(segments, delays, occ)


__all__ = [
    "parse_gtfs_static",
    "parse_gtfs_realtime",
    "parse_vehicle_positions",
    "build_snapshots",
    "load_gtfs",
]
