import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cost_gformer.gtfs import load_gtfs

from google.transit import gtfs_realtime_pb2


def _make_sample_gtfs(root: Path) -> None:
    (root / "stops.txt").write_text("stop_id,stop_name\nA,Stop A\nB,Stop B\nC,Stop C\n")

    with open(root / "trips.txt", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["route_id", "service_id", "trip_id"])
        w.writerow(["R1", "S1", "T1"])

    with open(root / "stop_times.txt", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"])
        w.writerow(["T1", "08:00:00", "08:00:00", "A", 1])
        w.writerow(["T1", "08:10:00", "08:10:00", "B", 2])
        w.writerow(["T1", "08:20:00", "08:20:00", "C", 3])

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    ent = feed.entity.add()
    ent.id = "1"
    tu = ent.trip_update
    tu.trip.trip_id = "T1"
    stu = tu.stop_time_update.add()
    stu.stop_sequence = 1
    stu.departure.delay = 30
    stu2 = tu.stop_time_update.add()
    stu2.stop_sequence = 2
    stu2.departure.delay = 60
    with open(root / "rt.pb", "wb") as f:
        f.write(feed.SerializeToString())


def test_load_gtfs(tmp_path: Path) -> None:
    _make_sample_gtfs(tmp_path)
    dataset = load_gtfs(str(tmp_path), str(tmp_path / "rt.pb"))
    assert len(dataset) == 2

    snap0 = dataset[0]
    assert snap0.time == 8 * 3600
    assert snap0.edges == [(0, 1)]
    assert float(snap0.static_edge_feat[(0, 1)][0]) == 600.0
    assert float(snap0.dynamic_edge_feat[(0, 1)][0]) == 30.0

    snap1 = dataset[1]
    assert snap1.time == 8 * 3600 + 10 * 60
    assert snap1.edges == [(1, 2)]
    assert float(snap1.static_edge_feat[(1, 2)][0]) == 600.0
    assert float(snap1.dynamic_edge_feat[(1, 2)][0]) == 60.0
