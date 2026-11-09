"""Kiểm thử baseline, tiêu chí so sánh, Golden Section và các chỉ số phụ. Chạy: python -m unittest discover -s tests -v"""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train import (baselines, confidence_threshold, ensure_not_tested, final_masks, golden_leaf,  # noqa: E402
                   is_better, rain_binary_accuracy, run, run_final_only, score)
import json  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
import tempfile  # noqa: E402


def small_frame(n_days=60, start='2020-01-01', seed=0):
    """X giả có index (origin_date, city) và các cột mà baseline cần."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, periods=n_days)
    idx = pd.MultiIndex.from_product([days, ['A', 'B']], names=['origin_date', 'city'])
    X = pd.DataFrame({'city': idx.get_level_values('city'),
                      'code_today': rng.integers(0, 3, len(idx)).astype(float),
                      'max': rng.normal(30, 2, len(idx)), 'min': rng.normal(22, 2, len(idx)),
                      'rain': rng.exponential(3, len(idx)), 'humidi': rng.normal(80, 5, len(idx))}, index=idx)
    return X


class BaselineTest(unittest.TestCase):
    def test_persistence_baseline_equals_today(self):
        X = small_frame()
        y = pd.Series(np.zeros(len(X)), index=X.index)
        for target, col in [('weather', 'code_today'), ('tmax', 'max'), ('tmin', 'min'), ('rain', 'rain')]:
            pred = baselines(X, y, X, target, 1)['persistence']
            np.testing.assert_array_equal(pred, X[col].to_numpy())

    def test_monthly_baseline_learned_from_train_only(self):
        X_tr = small_frame(n_days=31, start='2020-01-01')
        # Ngày cần dự báo (gốc + 1) của thành phố A trong tháng 1 phần lớn là Mưa (2), của B là Nắng (0).
        y_tr = pd.Series(np.where(X_tr.city == 'A', 2.0, 0.0), index=X_tr.index)
        X_ev = small_frame(n_days=10, start='2021-01-05')
        pred = baselines(X_tr, y_tr, X_ev, 'weather', 1)['monthly']
        np.testing.assert_array_equal(pred, np.where(X_ev.city == 'A', 2.0, 0.0))

    def test_monthly_baseline_mean_for_numbers_and_fallback(self):
        X_tr = small_frame(n_days=20, start='2020-03-01')
        y_tr = pd.Series(np.where(X_tr.city == 'A', 31.0, 27.0), index=X_tr.index)
        X_ev = small_frame(n_days=3, start='2021-03-05').iloc[:2]
        X_ev_unseen = small_frame(n_days=3, start='2021-08-05').iloc[:2]  # tháng 8 không có trong Train
        np.testing.assert_allclose(baselines(X_tr, y_tr, X_ev, 'tmax', 1)['monthly'], [31.0, 27.0])
        np.testing.assert_allclose(baselines(X_tr, y_tr, X_ev_unseen, 'tmax', 1)['monthly'], [29.0, 29.0])


class ScoreTest(unittest.TestCase):
    def test_score_keys(self):
        self.assertEqual(set(score('weather', [0, 1, 2], [0, 1, 1])), {'accuracy', 'macro_f1'})
        self.assertEqual(score('tmax', [30.0, 32.0], [31.0, 31.0]), {'mae': 1.0})

    def test_score_higher_is_better_flag(self):
        self.assertTrue(is_better('weather', 0.80, 0.70))
        self.assertFalse(is_better('weather', 0.70, 0.80))
        self.assertTrue(is_better('tmax', 1.2, 1.5))
        self.assertFalse(is_better('rain', 5.0, 4.0))


class GoldenTest(unittest.TestCase):
    def test_golden_leaf_returns_integer_in_bounds(self):
        X = small_frame(n_days=80)
        y = pd.Series((X['humidi'] > 80).astype(int).to_numpy(), index=X.index)
        tr, va = X.index.get_level_values(0) < '2020-03-01', X.index.get_level_values(0) >= '2020-03-01'
        result = golden_leaf('weather', {'n_estimators': 10}, X[tr], y[tr], X[va], y[va])
        self.assertIsInstance(result['leaf'], int)
        self.assertTrue(1 <= result['leaf'] <= 80)
        self.assertEqual(set(result), {'leaf', 'score', 'n_evals', 'curve'})
        self.assertEqual(result['n_evals'], len(result['curve']))


class ExtraMetricTest(unittest.TestCase):
    def test_rain_binary_accuracy(self):
        self.assertAlmostEqual(rain_binary_accuracy([2, 2, 1], [2, 1, 0]), 2 / 3)

    def test_confidence_threshold_keeps_70pct(self):
        proba = np.random.default_rng(1).uniform(0.34, 1.0, 100)
        kept = (proba >= confidence_threshold(proba)).sum()
        self.assertTrue(69 <= kept <= 71, kept)


class FinalFitTest(unittest.TestCase):
    def test_final_masks_cover_train_and_validation_only(self):
        split = pd.Series(['train', 'train', 'validation', 'test', 'test'])
        fit, test = final_masks(split)
        self.assertEqual(fit.tolist(), [True, True, True, False, False])
        self.assertEqual(test.tolist(), [False, False, False, True, True])

    def test_test_set_scored_only_once(self):
        with tempfile.TemporaryDirectory() as folder:
            run_dir = Path(folder)
            ensure_not_tested(run_dir)  # chưa có models/: được phép
            (run_dir / 'models').mkdir()
            (run_dir / 'models' / 'weather_h1.joblib').write_bytes(b'x')
            with self.assertRaisesRegex(ValueError, 'Test'):
                ensure_not_tested(run_dir)


class QuickRunTest(unittest.TestCase):
    def test_quick_run_never_scores_test(self):
        """--quick là bản thử nhanh: không được fit mô hình cuối hay chấm Test."""
        data = Path(__file__).resolve().parent.parent / 'data' / 'weather.csv'
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()) as out:
            run(data, Path(folder) / 'q', quick=True)
            self.assertFalse((Path(folder) / 'q' / 'models').exists())
            self.assertFalse((Path(folder) / 'q' / 'results' / 'test_metrics.csv').exists())
        self.assertNotIn('Test (20%', out.getvalue())


class TargetsTest(unittest.TestCase):
    def test_run_with_targets_trains_and_saves_only_those(self):
        """Cách tổng hợp: chỉ train các mô hình con số trên Open-Meteo."""
        data = Path(__file__).resolve().parent.parent / 'data' / 'open_meteo.csv'
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            out = Path(folder) / 'om'
            run(data, out, quick=True, targets=['tmax'])
            selected = json.loads((out / 'results' / 'selected_params.json').read_text(encoding='utf-8'))
            self.assertEqual(sorted(selected), ['tmax_h1', 'tmax_h2', 'tmax_h3'])
            run_final_only(data, out)
            self.assertEqual(sorted(p.name for p in (out / 'models').glob('*.joblib')),
                             ['tmax_h1.joblib', 'tmax_h2.joblib', 'tmax_h3.joblib'])
            meta = json.loads((out / 'models' / 'metadata.json').read_text(encoding='utf-8'))
            self.assertEqual((meta['targets'], meta['data_file']), (['tmax'], 'open_meteo.csv'))


if __name__ == '__main__':
    unittest.main()
