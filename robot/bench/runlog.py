"""
runlog.py — the bench's file format, defined once.

One trajectory = one csv. A `#` header carries JSON metadata, then ordinary
columns. Written by sweep.py, read by fit_bam.py, and generated synthetically by
`fit_bam.py --selftest`, so the format is exercised from both ends on every run.

The metadata is not decoration. A fit is only valid for the control registers the
servo had while the data was taken — change P_COEF or the dead zone and the
inner loop is a different machine — so every register in
registers.CONTROL_REGISTERS is stamped into the file and fit_bam.py refuses to
merge runs that disagree. The same goes for the load: the gravity torque is the
known quantity the whole identification is anchored to, so mass and radius are
part of the record, not a note in a lab book.
"""
from __future__ import annotations

import csv
import json
import os
import time

COLUMNS = ["t", "target_rad", "q_rad", "w_rad_s", "current_a",
           "volt_v", "temp_c", "load_raw", "counts"]


def write(path: str, meta: dict, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    meta = dict(meta)
    meta.setdefault("written", time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(path, "w", newline="") as f:
        for line in json.dumps(meta, indent=1, sort_keys=True).splitlines():
            f.write(f"# {line}\n")
        w = csv.DictWriter(f, COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read(path: str):
    """-> (meta, {column: list})"""
    head, body = [], []
    with open(path) as f:
        for line in f:
            (head if line.startswith("#") else body).append(line)
    meta = json.loads("".join(l[2:] if l.startswith("# ") else l[1:] for l in head))
    cols = {c: [] for c in COLUMNS}
    for row in csv.DictReader(body):
        for c in COLUMNS:
            v = row.get(c, "")
            cols[c].append(float(v) if v not in ("", None) else float("nan"))
    return meta, cols


def load_dir(d: str, pattern: str = ".csv"):
    out = []
    for name in sorted(os.listdir(d)):
        if name.endswith(pattern):
            out.append((name, *read(os.path.join(d, name))))
    return out
