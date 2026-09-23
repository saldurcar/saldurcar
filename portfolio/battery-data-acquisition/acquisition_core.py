"""Sanitized portfolio example of a battery experiment acquisition core.

This is a hardware-neutral version derived from a larger laboratory acquisition
workflow. Device-specific ports, camera drivers and private experiment paths are
intentionally omitted.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import csv
import math
import time


@dataclass
class SensorSample:
    timestamp: float
    voltage_v: float
    current_ma: float
    temp1_c: float
    temp2_c: float
    temp3_c: float


def finite_or_none(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def parse_controller_line(line):
    """Parse a generic comma-separated controller record.

    Expected public demo format:
        temp1,temp2,temp3,voltage,current_mA
    """
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 5:
        raise ValueError("Expected five comma-separated values")

    t1, t2, t3, voltage, current = map(float, parts)
    return SensorSample(
        timestamp=time.time(),
        voltage_v=voltage,
        current_ma=current,
        temp1_c=t1,
        temp2_c=t2,
        temp3_c=t3,
    )


class AcquisitionBuffer:
    """Rolling sensor buffer with simple thermal-rate estimation."""

    def __init__(self, maxlen=300):
        self.samples = deque(maxlen=maxlen)

    def append(self, sample):
        self.samples.append(sample)

    def latest_dtdt(self, channel="temp1_c"):
        if len(self.samples) < 2:
            return 0.0

        a, b = self.samples[-2], self.samples[-1]
        dt = b.timestamp - a.timestamp
        if dt <= 0:
            return 0.0

        return (getattr(b, channel) - getattr(a, channel)) / dt

    def latest(self):
        return self.samples[-1] if self.samples else None


def append_csv(path, sample, dtdt):
    """Append one synchronized record to a CSV file."""
    row = {
        "timestamp_iso": datetime.fromtimestamp(sample.timestamp).isoformat(
            timespec="milliseconds"
        ),
        "voltage_v": sample.voltage_v,
        "current_ma": sample.current_ma,
        "temp1_c": sample.temp1_c,
        "temp2_c": sample.temp2_c,
        "temp3_c": sample.temp3_c,
        "dtemp1_dt_c_per_s": dtdt,
    }

    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
