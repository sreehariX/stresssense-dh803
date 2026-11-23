import unittest

import numpy as np
import pandas as pd

from stresssense.datasets import physionet_segments, wesad_segments
from stresssense.evaluate import hold_out_participants, scores
from stresssense.signals import RATES, Segment, Session, Signal, extract_features, window_times


class TestWindowPreparation(unittest.TestCase):
    def test_wesad_uses_only_source_labels_one_and_two(self):
        data = np.repeat([0, 1, 6, 2, 3], [700, 1400, 700, 700, 700])
        self.assertEqual(list(wesad_segments(data)), [
            Segment(1, 3, 0, "baseline"),
            Segment(4, 5, 1, "tsst"),
        ])

    def test_physionet_protocol_versions_and_split_session(self):
        v1 = list(physionet_segments(list(range(13)), "V1"))
        v2 = list(physionet_segments(list(range(9)), "V2", has_baseline=False))
        self.assertIn(Segment(2, 3, 1, "stroop"), v1)
        self.assertIn(Segment(5, 6, 0, "rest_2"), v1)
        self.assertIn(Segment(1, 2, 1, "mental_arithmetic"), v2)
        self.assertIn(Segment(6, 7, 0, "rest_2"), v2)
        self.assertFalse(any(segment.task == "baseline" for segment in v2))
        with self.assertRaises(ValueError):
            list(physionet_segments([0, 1, 2, 3, 4, 5, 6, 7, 7], "V2"))

    def test_windows_fit_inside_all_sensors_and_label_guard(self):
        signals = {"BVP": Signal(np.zeros(90 * 64), 64, 10),
                   "EDA": Signal(np.zeros(70 * 4), 4, 12)}
        segment = Segment(4, 90, 0, "baseline")
        session = Session("test", "test", signals, [segment])
        self.assertEqual(list(window_times(session)), [(12, segment), (42, segment)])

    def test_pulse_rate_on_known_wave_and_constant_signal(self):
        seconds = np.arange(30 * RATES["BVP"]) / RATES["BVP"]
        window = {
            "BVP": np.sin(2 * np.pi * 1.25 * seconds),
            "EDA": np.ones(120), "TEMP": np.full(120, 33.0),
            "ACC": np.tile([0, 0, 1], (960, 1)),
        }
        features = extract_features(window)
        self.assertAlmostEqual(features["bvp_rate"], 75.0, delta=0.5)
        self.assertEqual(features["acc_axis_std"], 0.0)
        window["BVP"] *= 0
        self.assertTrue(np.isnan(extract_features(window)["bvp_rate"]))


class TestEvaluation(unittest.TestCase):
    def test_held_out_rows_covered_once(self):
        data = pd.DataFrame({
            "participant": np.repeat(["A", "B", "C", "D"], 2),
            "cohort": "synthetic",
            "start_s": [0, 30] * 4,
            "task": ["baseline", "stress"] * 4,
            "label": [0, 1] * 4,
            "eda_mean": [1, 5, 2, 6, 3, 7, 1, 8],
        })
        result = hold_out_participants(data, "logistic", ["eda_mean"])
        self.assertEqual(result.index.to_list(), data.index.to_list())
        self.assertEqual(result.prediction.to_list(), data.label.to_list())

    def test_metrics_apply_equal_weight_to_classes(self):
        result = scores(tn=90, fp=0, fn=9, tp=1)
        self.assertAlmostEqual(result["balanced_accuracy"], 0.55)
        self.assertAlmostEqual(result["stress_recall"], 0.10)
        self.assertAlmostEqual(result["accuracy"], 0.91)


if __name__ == "__main__":
    unittest.main()
