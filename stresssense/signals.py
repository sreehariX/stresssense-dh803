"""Window timing and features shared by both datasets."""

from dataclasses import dataclass
from math import floor

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


RATES = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32}
WINDOW = 30
GUARD = 5


@dataclass(frozen=True)
class Signal:
    samples: np.ndarray
    rate: int
    start: float = 0.0

    @property
    def end(self):
        return self.start + len(self.samples) / self.rate


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    label: int
    task: str


@dataclass(frozen=True)
class Session:
    participant: str
    cohort: str
    signals: dict[str, Signal]
    segments: list[Segment]


def window_times(session):
    signal_start = max(signal.start for signal in session.signals.values())
    signal_end = min(signal.end for signal in session.signals.values())
    for segment in session.segments:
        first = max(segment.start + GUARD, signal_start)
        last = min(segment.end - GUARD, signal_end)
        count = floor((last - first + 1e-8) / WINDOW)
        for index in range(count):
            yield first + index * WINDOW, segment


def slice_window(signal, start):
    # Start at the first sample at or after the requested time.
    index = int(np.ceil((start - signal.start) * signal.rate - 1e-5))
    end = index + WINDOW * signal.rate
    if index < 0 or end > len(signal.samples):
        raise ValueError("A requested window falls outside the recording")
    return signal.samples[index:end]


def _slope(values, rate):
    time = np.arange(len(values)) / rate
    time -= time.mean()
    return float((time @ values) / (time @ time))


def _stats(values, rate, lower, upper):
    good = np.isfinite(values) & (values >= lower) & (values <= upper)
    if good.mean() < 0.90:
        return dict.fromkeys(("mean", "std", "slope", "range"), np.nan)
    indices = np.arange(len(values))
    values = np.interp(indices, indices[good], values[good])
    return {
        "mean": values.mean(),
        "std": values.std(),
        "slope": _slope(values, rate),
        "range": np.quantile(values, 0.9) - np.quantile(values, 0.1),
    }


BVP_FILTER = butter(3, (0.6, 4.0), btype="bandpass", fs=RATES["BVP"], output="sos")


def _pulse(values):
    missing = dict.fromkeys(("rate", "interval_iqr"), np.nan)
    if not np.isfinite(values).all() or np.ptp(values) < 1e-6:
        return missing
    rate = RATES["BVP"]
    filtered = sosfiltfilt(BVP_FILTER, values)
    spread = np.quantile(filtered, 0.9) - np.quantile(filtered, 0.1)
    peaks, _ = find_peaks(filtered, distance=int(0.30 * rate), prominence=0.30 * spread)
    peaks = peaks[(peaks >= rate) & (peaks < len(values) - rate)]
    intervals = np.diff(peaks) / rate
    valid = intervals[(intervals >= 0.33) & (intervals <= 1.50)]
    if len(valid) < 12 or len(valid) < 0.75 * len(intervals) or valid.sum() < 18:
        return missing
    return {
        "rate": 60 / np.median(valid),
        "interval_iqr": np.quantile(valid, 0.75) - np.quantile(valid, 0.25),
    }


def extract_features(window):
    acc = window["ACC"]
    motion = dict.fromkeys(("magnitude_std", "axis_std", "jerk"), np.nan)
    if np.isfinite(acc).all():
        motion = {
            "magnitude_std": np.linalg.norm(acc, axis=1).std(),
            "axis_std": acc.std(axis=0).mean(),
            "jerk": np.linalg.norm(np.diff(acc, axis=0), axis=1).mean() * RATES["ACC"],
        }
    blocks = {
        "bvp": _pulse(window["BVP"]),
        "eda": _stats(window["EDA"], RATES["EDA"], 0, 100),
        "temp": _stats(window["TEMP"], RATES["TEMP"], 20, 45),
        "acc": motion,
    }
    return {f"{sensor}_{name}": value for sensor, values in blocks.items() for name, value in values.items()}


def recording_window(session, start):
    return {name: slice_window(signal, start) for name, signal in session.signals.items()}


def feature_rows(session):
    origin = min(signal.start for signal in session.signals.values())
    for start, segment in window_times(session):
        yield {
            "participant": session.participant,
            "cohort": session.cohort,
            "start_s": start - origin,
            "label": segment.label,
            "task": segment.task,
            **extract_features(recording_window(session, start)),
        }
