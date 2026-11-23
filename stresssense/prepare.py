"""Extract window features and save an audit of the recordings used."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .datasets import PHYSIONET_EXCLUSIONS, PHYSIONET_IDS, WESAD_IDS, physionet_sessions, wesad_sessions
from .signals import GUARD, WINDOW, feature_rows


def _sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def prepare(wesad, physionet, output):
    output.mkdir(parents=True, exist_ok=True)
    audit = {"window_seconds": WINDOW, "edge_guard_seconds": GUARD}
    sources = (
        ("wesad", wesad, wesad_sessions, WESAD_IDS),
        ("physionet", physionet, physionet_sessions, PHYSIONET_IDS - PHYSIONET_EXCLUSIONS.keys()),
    )
    for name, archive, loader, expected in sources:
        rows, durations = [], []
        for session in loader(archive):
            part = list(feature_rows(session))
            if {row["label"] for row in part} != {0, 1}:
                raise ValueError(f"{name} {session.participant} does not have windows from both classes")
            rows.extend(part)
            durations.append({"participant": session.participant, "cohort": session.cohort,
                              "rest": sum(row["label"] == 0 for row in part),
                              "stress": sum(row["label"] == 1 for row in part)})
            print(f"{name} {session.participant}: {len(part)} windows", flush=True)
        frame = pd.DataFrame(rows)
        if set(frame.participant) != expected or frame.duplicated(["participant", "start_s"]).any():
            raise ValueError(f"Unexpected or duplicate feature rows for {name}")
        feature_columns = frame.filter(regex=r"^(acc|bvp|eda|temp)_")
        if feature_columns.isna().all(axis=1).any():
            raise ValueError(f"A window has no usable features in {name}")
        frame.to_csv(output / f"{name}_features.csv", index=False)
        pd.DataFrame(durations).to_csv(output / f"{name}_participant_windows.csv", index=False)
        audit[name] = {
            "archive_sha256": _sha256(archive),
            "participants": len(expected),
            "windows": len(frame),
            "rest_windows": int((frame.label == 0).sum()),
            "stress_windows": int((frame.label == 1).sum()),
            "task_windows": frame.task.value_counts().sort_index().to_dict(),
            "missing_by_feature": feature_columns.isna().sum().to_dict(),
        }
    audit["physionet"]["excluded_participants"] = PHYSIONET_EXCLUSIONS
    audit["physionet"]["partial_participants"] = {"f14": "Tagged f14_b only. Untagged baseline in f14_a omitted."}
    (output / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wesad", type=Path, default=Path("data/WESAD.zip"))
    parser.add_argument("--physionet", type=Path, default=Path("data/physionet-1.0.1.zip"))
    parser.add_argument("--output", type=Path, default=Path("run"))
    args = parser.parse_args()
    prepare(args.wesad, args.physionet, args.output)


if __name__ == "__main__":
    main()
