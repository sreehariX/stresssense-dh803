"""Evaluate fixed models on unseen people, then on an independent study."""

import argparse
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 803
FEATURE_GROUPS = ("bvp", "eda", "temp", "acc")
PRIMARY = "logistic"
MODELS = ("always_rest", PRIMARY, "random_forest", "acc_only",
          "without_bvp", "without_eda", "without_temp", "without_acc")
METRICS = ("balanced_accuracy", "stress_f1", "stress_recall", "rest_recall", "accuracy")
BOOTSTRAP_REPEATS = 5000


def feature_columns(frame, model):
    sensors = FEATURE_GROUPS
    if model == "acc_only":
        sensors = ("acc",)
    elif model.startswith("without_"):
        sensors = tuple(sensor for sensor in sensors if model != f"without_{sensor}")
    return [column for column in frame if column.partition("_")[0] in sensors]


def classifier(model):
    if model == "always_rest":
        return DummyClassifier(strategy="constant", constant=0)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    if model == "random_forest":
        forest = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                        class_weight="balanced", random_state=SEED, n_jobs=1)
        return make_pipeline(imputer, forest)
    return make_pipeline(imputer, StandardScaler(),
                         LogisticRegression(C=1, class_weight="balanced", max_iter=2000,
                                            random_state=SEED))


def _predict(estimator, columns, test):
    results = test[["participant", "cohort", "start_s", "task", "label"]].copy()
    results["stress_score"] = estimator.predict_proba(test[columns])[:, 1]
    results["prediction"] = (results["stress_score"] > 0.5).astype(int)
    return results


def hold_out_participants(wesad, model, columns):
    folds = []
    for train_index, test_index in LeaveOneGroupOut().split(wesad, groups=wesad.participant):
        train = wesad.iloc[train_index]
        test = wesad.iloc[test_index]
        if test.participant.nunique() != 1 or set(train.participant) & set(test.participant):
            raise AssertionError("Participant leakage in WESAD split")
        estimator = classifier(model).fit(train[columns], train.label)
        folds.append(_predict(estimator, columns, test))
    results = pd.concat(folds).sort_index()
    if not results.index.equals(wesad.index):
        raise AssertionError("Each WESAD row must have exactly one held-out prediction")
    return results


def counts(frame):
    return {
        "tn": int(((frame.label == 0) & (frame.prediction == 0)).sum()),
        "fp": int(((frame.label == 0) & (frame.prediction == 1)).sum()),
        "fn": int(((frame.label == 1) & (frame.prediction == 0)).sum()),
        "tp": int(((frame.label == 1) & (frame.prediction == 1)).sum()),
    }


def scores(tn, fp, fn, tp):
    if not (tp + fn and tn + fp):
        raise ValueError("Both classes are required for participant metrics")
    rest_recall = tn / (tn + fp)
    stress_recall = tp / (tp + fn)
    return {
        "balanced_accuracy": (stress_recall + rest_recall) / 2,
        "stress_f1": (2 * tp) / (2 * tp + fp + fn),
        "stress_recall": stress_recall,
        "rest_recall": rest_recall,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
    }


def summarize(frame):
    people = []
    for person, group in frame.groupby("participant"):
        units = counts(group)
        people.append({"participant": person, "cohort": group.cohort.iloc[0], "windows": len(group),
                       **units, **scores(**units)})
    participant_scores = pd.DataFrame(people)
    samples = participant_scores[list(METRICS)].to_numpy()
    rng = np.random.default_rng(SEED)
    resamples = rng.integers(len(samples), size=(BOOTSTRAP_REPEATS, len(samples)))
    lower, upper = np.quantile(samples[resamples].mean(axis=1), (0.025, 0.975), axis=0)
    summary = {
        "participants": len(people),
        "windows": len(frame),
        "pooled_confusion": counts(frame),
        "mean": dict(zip(METRICS, samples.mean(axis=0).tolist())),
        "bootstrap_95": dict(zip(METRICS, np.column_stack((lower, upper)).tolist())),
    }
    return summary, participant_scores


def experiment(features, output):
    output.mkdir(parents=True, exist_ok=True)
    wesad = pd.read_csv(features / "wesad_features.csv")
    columns_by_model = {model: feature_columns(wesad, model) for model in MODELS}
    fits, internal = {}, {}
    for model, columns in columns_by_model.items():
        internal[model] = hold_out_participants(wesad, model, columns)
        fits[model] = classifier(model).fit(wesad[columns], wesad.label)
        print(f"WESAD evaluation complete for {model}", flush=True)
    # Persist the fixed source model before opening the target feature table.
    joblib.dump({"estimator": fits[PRIMARY], "features": columns_by_model[PRIMARY], "threshold": 0.5},
                output / "logistic_wesad.joblib")
    external = pd.read_csv(features / "physionet_features.csv")
    summary = {
        "primary_model": PRIMARY,
        "seed": SEED,
        "threshold": 0.5,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
        "models": {},
    }
    predictions, participant_metrics = [], []
    for model, columns in columns_by_model.items():
        summary["models"][model] = {}
        comparisons = {"wesad_loso": internal[model], "physionet_transfer": _predict(fits[model], columns, external)}
        for evaluation, frame in comparisons.items():
            aggregate, people = summarize(frame)
            summary["models"][model][evaluation] = aggregate
            predictions.append(frame.assign(model=model, evaluation=evaluation))
            participant_metrics.append(people.assign(model=model, evaluation=evaluation))
        if model == PRIMARY:
            target = comparisons["physionet_transfer"]
            summary["physionet_cohorts"] = {
                cohort: summarize(group)[0] for cohort, group in target.groupby("cohort")
            }
            baseline_ids = set(target.loc[target.task == "baseline", "participant"])
            baseline_compare = target[target.participant.isin(baseline_ids) &
                                      ((target.label == 1) | (target.task == "baseline"))]
            summary["physionet_baseline_only"] = summarize(baseline_compare)[0]
            summary["physionet_tasks"] = {
                task: {"windows": len(part), "participants": int(part.participant.nunique()),
                       "correct": int(part.label.eq(part.prediction).sum())}
                for task, part in target.groupby("task")
            }
    pd.concat(predictions, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    pd.concat(participant_metrics, ignore_index=True).to_csv(output / "participant_metrics.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for name, result in summary["models"][PRIMARY].items():
        print(f"{name}: {result['participants']} people, balanced accuracy {result['mean']['balanced_accuracy']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("run"))
    parser.add_argument("--output", type=Path, default=Path("run"))
    args = parser.parse_args()
    experiment(args.features, args.output)


if __name__ == "__main__":
    main()
