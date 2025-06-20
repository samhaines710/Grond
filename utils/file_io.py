# utils/file_io.py

import os
import json
import csv
from datetime import datetime
from config import SIGNAL_TRACKER_FILE, SNAPSHOT_FILE
from logging_utils import write_status

def append_signal_log(signal: dict):
    """
    Append a trade signal to daily CSV.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    fname    = SIGNAL_TRACKER_FILE.replace(".csv", f"_{date_str}.csv")
    fields   = list(signal.keys())
    os.makedirs(os.path.dirname(fname) or ".", exist_ok=True)
    exists = os.path.isfile(fname)
    with open(fname, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(signal)

def load_snapshot() -> dict:
    """
    Load previous snapshot from disk.
    """
    if os.path.exists(SNAPSHOT_FILE):
        with open(SNAPSHOT_FILE) as f:
            return json.load(f)
    return {}

def save_snapshot(snapshot: dict):
    """
    Overwrite the current snapshot on disk.
    """
    os.makedirs(os.path.dirname(SNAPSHOT_FILE) or ".", exist_ok=True)
    with open(SNAPSHOT_FILE, "w") as f:
        json.dump(snapshot, f, indent=2)
    write_status("Saved new options snapshot.")
