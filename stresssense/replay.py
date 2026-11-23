"""Create an offline browser replay for one held-out participant."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import physionet_sessions, wesad_sessions
from .evaluate import PRIMARY
from .signals import RATES, recording_window


def replay(dataset, participant, archive, predictions, output):
    code = "wesad_loso" if dataset == "wesad" else "physionet_transfer"
    frame = pd.read_csv(predictions)
    rows = frame[(frame.participant == participant) & (frame.evaluation == code) &
                 (frame.model == PRIMARY)].sort_values("start_s")
    if rows.empty:
        raise ValueError(f"No held-out predictions for {participant} in {dataset}")
    loader = wesad_sessions if dataset == "wesad" else physionet_sessions
    session = next(loader(archive, participant))
    origin = min(signal.start for signal in session.signals.values())
    windows = []
    for row in rows.itertuples():
        signals = recording_window(session, origin + row.start_s)
        traces = {}
        for name, trace in signals.items():
            stride = {"BVP": 4, "ACC": 2}.get(name, 1)
            values = np.linalg.norm(trace, axis=1) if name == "ACC" else trace
            traces[name] = [round(float(v), 4) if np.isfinite(v) else None for v in values[::stride]]
        windows.append({
            "start": round(row.start_s, 2),
            "task": row.task,
            "label": row.label,
            "prediction": row.prediction,
            "score": round(row.stress_score, 5),
            "traces": traces,
        })
    payload = {
        "study": "WESAD" if dataset == "wesad" else "PhysioNet",
        "participant": participant,
        "windows": windows,
        "rates": {name: rate // {"BVP": 4, "ACC": 2}.get(name, 1) for name, rate in RATES.items()},
    }
    template = Path(__file__).with_name("replay.html").read_text(encoding="utf-8")
    content = template.replace("__STRESSSENSE_DATA__", json.dumps(payload, allow_nan=False).replace("<", "\\u003c"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(f"{output}: {len(windows)} evaluated windows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["wesad", "physionet"])
    parser.add_argument("participant")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--predictions", type=Path, default=Path("run/predictions.csv"))
    parser.add_argument("--output", type=Path, default=Path("run/replay.html"))
    args = parser.parse_args()
    replay(args.dataset, args.participant, args.archive, args.predictions, args.output)


if __name__ == "__main__":
    main()
