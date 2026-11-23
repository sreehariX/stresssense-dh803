"""Load the original WESAD and PhysioNet archives without unpacking them."""

from datetime import datetime, timezone
import io
import pickle
from pathlib import PurePosixPath
from zipfile import ZipFile

import numpy as np

from .signals import RATES, Segment, Session, Signal


WESAD_IDS = {f"S{i}" for i in range(2, 18) if i != 12}
PHYSIONET_IDS = {f"S{i:02}" for i in range(1, 19)} | {f"f{i:02}" for i in range(1, 19)}
PHYSIONET_EXCLUSIONS = {
    "S02": "Duplicated raw signals documented by source",
    "f07": "Protective dock covered the pulse and temperature sensors",
}


class _WesadUnpickler(pickle.Unpickler):
    # WESAD stores NumPy arrays in Python 2 pickle files. Reject unrelated globals.
    _allowed = {
        ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
        ("_codecs", "encode"): __import__("codecs").encode,
    }

    def find_class(self, module, name):
        try:
            return self._allowed[module, name]
        except KeyError as error:
            raise pickle.UnpicklingError(f"Unexpected pickle global {module}.{name}") from error


def wesad_segments(labels):
    label_rate = 700
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    edges = np.r_[0, changes, len(labels)]
    for start, end in zip(edges[:-1], edges[1:]):
        source_label = labels[start]
        if source_label in (1, 2):
            yield Segment(start / label_rate, end / label_rate, int(source_label == 2),
                          "baseline" if source_label == 1 else "tsst")


def _wesad_members(archive):
    members = {}
    for name in archive.namelist():
        path = PurePosixPath(name)
        if path.suffix == ".pkl" and path.parent.name == path.stem and path.stem in WESAD_IDS:
            if path.stem in members:
                raise ValueError(f"Duplicate participant in WESAD archive: {path.stem}")
            members[path.stem] = name
    if members.keys() != WESAD_IDS:
        raise ValueError(f"Missing WESAD participants: {sorted(WESAD_IDS - members.keys())}")
    return members


def wesad_sessions(path, participant=None):
    with ZipFile(path) as archive:
        members = _wesad_members(archive)
        ids = [participant] if participant else sorted(members, key=lambda key: int(key[1:]))
        for person in ids:
            with archive.open(members[person]) as stream:
                data = _WesadUnpickler(stream, encoding="latin1").load()
            if data["subject"] != person:
                raise ValueError(f"Unexpected participant identifier in {members[person]}")
            wrist = data["signal"]["wrist"]
            signals = {}
            for name, rate in RATES.items():
                samples = np.asarray(wrist[name], dtype=float)
                samples = samples / 64 if name == "ACC" else samples.ravel()
                signals[name] = Signal(samples, rate)
            segments = list(wesad_segments(data["label"].ravel()))
            del data, wrist
            yield Session(person, "WESAD", signals, segments)


# Zero-based row indices in the raw tags.csv. The authors' notebook prepends
# the session origin, so its index is one higher than the corresponding row.
PHYSIONET_TAGS = {
    "V1": (
        (0, 1, 0, "baseline"), (2, 3, 1, "stroop"), (3, 4, 0, "rest_1"),
        (4, 5, 1, "mental_arithmetic"), (5, 6, 0, "rest_2"),
        (6, 7, 1, "opinion"), (8, 9, 1, "counter_opinion"), (10, 11, 1, "subtraction"),
    ),
    "V2": (
        (0, 1, 0, "baseline"), (1, 2, 1, "mental_arithmetic"), (2, 3, 0, "rest_1"),
        (3, 4, 1, "opinion"), (5, 6, 1, "counter_opinion"), (6, 7, 0, "rest_2"),
        (7, 8, 1, "subtraction"),
    ),
}


def _utc(text):
    value = datetime.fromisoformat(text.strip())
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


def physionet_segments(tags, cohort, has_baseline=True):
    expected = {"V1": 13, "V2": 9}[cohort]
    if len(tags) != expected or any(right <= left for left, right in zip(tags, tags[1:])):
        raise ValueError(f"Invalid PhysioNet {cohort} event times. Expected {expected} ordered tags.")
    for start, end, label, name in PHYSIONET_TAGS[cohort]:
        if name != "baseline" or has_baseline:
            yield Segment(tags[start], tags[end], label, name)


def _empatica_csv(archive, name, signal):
    with io.TextIOWrapper(archive.open(name), encoding="utf-8-sig") as stream:
        start = _utc(stream.readline().split(",")[0])
        rate = float(stream.readline().split(",")[0])
        if rate != RATES[signal]:
            raise ValueError(f"Unexpected sample rate {rate} in {name}")
        values = np.loadtxt(stream, delimiter=",", ndmin=2)
    if signal == "ACC" and values.shape[1] == 3:
        return Signal(values / 64, int(rate), start)
    if signal != "ACC" and values.shape[1] == 1:
        return Signal(values[:, 0], int(rate), start)
    raise ValueError(f"Unexpected signal shape {values.shape} in {name}")


def physionet_sessions(path, participant=None):
    with ZipFile(path) as archive:
        locations = {}
        for name in archive.namelist():
            parent = PurePosixPath(name).parent
            if parent.parent.name == "STRESS" and PurePosixPath(name).name == "tags.csv":
                locations[parent.name] = parent.as_posix()
        expected = (PHYSIONET_IDS - {"f14"}) | {"f14_a", "f14_b"}
        if locations.keys() != expected:
            raise ValueError(f"Unexpected PhysioNet directory structure: {sorted(locations.keys() ^ expected)}")
        ids = [participant] if participant else sorted(PHYSIONET_IDS - PHYSIONET_EXCLUSIONS.keys())
        for person in ids:
            if person in PHYSIONET_EXCLUSIONS:
                raise ValueError(f"{person} is excluded: {PHYSIONET_EXCLUSIONS[person]}")
            cohort = "V1" if person.startswith("S") else "V2"
            location = locations["f14_b" if person == "f14" else person]
            signals = {name: _empatica_csv(archive, f"{location}/{name}.csv", name) for name in RATES}
            origin = min(signal.start for signal in signals.values())
            signals = {name: Signal(s.samples, s.rate, s.start - origin) for name, s in signals.items()}
            events = archive.read(f"{location}/tags.csv").decode("utf-8-sig").splitlines()
            tags = [_utc(line) - origin for line in events if line.strip()]
            segments = list(physionet_segments(tags, cohort, has_baseline=person != "f14"))
            yield Session(person, cohort, signals, segments)
