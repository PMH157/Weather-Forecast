"""Kiểm thử bản tin dự báo. Chạy: python -m unittest discover -s tests -v"""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import HORIZONS, TARGETS, make_features, make_targets  # noqa: E402
from predict import (bulletin_row, configure_console, day_label, describe_run, forecast, list_runs, pick_run,  # noqa: E402
                     run_contents, runs_with,
                     print_bulletin)
from datetime import date  # noqa: E402
from test_features import wide_from_days  # noqa: E402
from train import make_pipeline, model_path  # noqa: E402
from evaluate_live import live_weather_accuracy  # noqa: E402

DAYS = pd.date_range('2021-01-01', '2021-02-28').strftime('%Y-%m-%d').tolist()


def noisy_wide(days, seed=0):
    wide = wide_from_days(days)
    rng = np.random.default_rng(seed)
    shape = wide['rain'].shape
    wide['rain'] = wide['rain'] + rng.exponential(2, shape)
    wide['cloud'] = wide['cloud'] + rng.uniform(-30, 50, shape)
    wide['max'] = wide['max'] + rng.normal(0, 2, shape)
    return wide


class PredictTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Một lần chạy giả: 12 RF nhỏ fit trên dữ liệu tự tạo, cùng định dạng thư mục với train.py."""
        cls.folder = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls.folder.name)
        models = cls.run_dir / 'models'
        models.mkdir()
        wide = noisy_wide(DAYS)
        for h in HORIZONS:
            X = make_features(wide, h)
            T = make_targets(wide, h).reindex(X.index)
            keep = T.notna().all(axis=1)
            for target in TARGETS:
                y = T.loc[keep, f'{target}_h{h}']
                y = y.astype(int) if target == 'weather' else y
                kind = 'clf' if target == 'weather' else 'reg'
                pipe = make_pipeline(kind, {'n_estimators': 5}).fit(X[keep], y)
                joblib.dump(pipe, model_path(models, target, h))
        meta = {'feature_columns': list(X.columns), 'confidence_threshold': {str(h): 0.5 for h in HORIZONS}}
        (models / 'metadata.json').write_text(json.dumps(meta), encoding='utf-8')
        # Lần chạy thứ hai chỉ có mô hình con số, học trên nhãn + 100 để nhận ra kết quả đến từ đâu.
        cls.numbers_run = cls.run_dir / 'numbers'
        (cls.numbers_run / 'models').mkdir(parents=True)
        for h in HORIZONS:
            X = make_features(wide, h)
            T = make_targets(wide, h).reindex(X.index)
            keep = T.notna().all(axis=1)
            for target in ['tmax', 'tmin', 'rain']:
                pipe = make_pipeline('reg', {'n_estimators': 5}).fit(X[keep], T.loc[keep, f'{target}_h{h}'] + 100)
                joblib.dump(pipe, model_path(cls.numbers_run / 'models', target, h))
        (cls.numbers_run / 'models' / 'metadata.json').write_text(json.dumps(
            {'feature_columns': list(X.columns), 'targets': ['tmax', 'tmin', 'rain'], 'data_file': 'open_meteo.csv'}),
            encoding='utf-8')

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_forecast_three_days_per_city(self):
        result = forecast(noisy_wide(DAYS, seed=1), self.run_dir)
        self.assertEqual(len(result), 9)
        self.assertEqual(sorted(result.groupby('city').h.apply(list).iloc[0]), [1, 2, 3])
        last = pd.Timestamp(DAYS[-1])
        self.assertTrue((result.target_date == last + pd.to_timedelta(result.h, unit='D')).all())

    def test_short_csv_still_forecasts(self):
        self.assertEqual(len(forecast(noisy_wide(DAYS[-4:], seed=2), self.run_dir)), 9)

    def test_city_missing_last_day_reported(self):
        wide = noisy_wide(DAYS, seed=3)
        for table in wide.values():
            table.loc[DAYS[-1], 'Nha Trang'] = np.nan
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            result = forecast(wide, self.run_dir)
        self.assertEqual(len(result), 6)
        self.assertNotIn('Nha Trang', set(result.city))
        self.assertIn('Nha Trang', err.getvalue())

    def test_rain_clipped_and_note(self):
        row = bulletin_row('Ha Noi', pd.Timestamp('2021-06-18'), 1, 2, 0.9, 31.0, 24.0, -0.3, 0.6)
        self.assertEqual(row['rain_mm'], 0.0)
        self.assertNotEqual(row['note'], '')
        self.assertEqual(row['confidence'], 'cao')
        self.assertEqual(bulletin_row('Ha Noi', pd.Timestamp('2021-06-18'), 1, 0, 0.4, 31, 24, 0.2, 0.6)['note'], '')

    def test_print_survives_cp1252(self):
        stream = io.TextIOWrapper(io.BytesIO(), encoding='cp1252', errors='strict')
        configure_console([stream])
        result = forecast(noisy_wide(DAYS, seed=4), self.run_dir)
        print_bulletin(result, stream)  # không được ném UnicodeEncodeError
        stream.flush()
        self.assertIn('Ha Noi', stream.buffer.getvalue().decode('utf-8'))


    def test_numbers_come_from_numbers_run_when_given(self):
        wide = noisy_wide(DAYS, seed=5)
        alone, combined = forecast(wide, self.run_dir), forecast(wide, self.run_dir, numbers_run=self.numbers_run)
        self.assertTrue((combined.tmax > 90).all() and (combined.rain_mm > 90).all())
        self.assertTrue((alone.tmax < 90).all())
        self.assertEqual(combined.weather_code.tolist(), alone.weather_code.tolist())


    def test_live_accuracy_same_source_gives_same_numbers(self):
        wide = noisy_wide(DAYS, seed=6)
        result = live_weather_accuracy(self.run_dir, wide, wide)
        self.assertEqual(result.h.tolist(), [1, 2, 3])
        self.assertTrue((result.days > 0).all())
        self.assertEqual(result.accuracy_kaggle_input.tolist(), result.accuracy_open_meteo_input.tolist())


    def test_missing_weather_model_gives_clear_error(self):
        """Chọn nhầm lần chạy chỉ có mô hình con số cho ô Loại thời tiết: báo lỗi tiếng Việt, không FileNotFoundError."""
        with self.assertRaisesRegex(ValueError, 'không có mô hình loại thời tiết'):
            forecast(noisy_wide(DAYS, seed=7), self.numbers_run)


class DayLabelTest(unittest.TestCase):
    def test_day_label_relative_to_today(self):
        today = date(2026, 11, 9)  # Thứ Hai
        labels = [day_label(pd.Timestamp(d), today) for d in ['2026-11-09', '2026-11-10', '2026-11-11', '2026-11-12']]
        self.assertEqual(labels, ['Hôm nay, Thứ Hai 09/11', 'Ngày mai, Thứ Ba 10/11', 'Ngày kia, Thứ Tư 11/11',
                                  'Thứ Năm 12/11'])

    def test_day_label_without_today(self):
        self.assertEqual(day_label(pd.Timestamp('2021-06-19')), 'Thứ Bảy 19/06')


class ListRunsTest(unittest.TestCase):
    def test_list_runs_only_finished_runs_newest_first(self):
        with tempfile.TemporaryDirectory() as folder:
            runs = Path(folder)
            for name, finished in [('20261101_100000', True), ('20261102_100000', False), ('20261103_100000', True)]:
                (runs / name / 'models').mkdir(parents=True)
                if finished:
                    (runs / name / 'models' / 'metadata.json').write_text('{}', encoding='utf-8')
            self.assertEqual([p.name for p in list_runs(runs)], ['20261103_100000', '20261101_100000'])

    def test_list_runs_missing_folder_is_empty(self):
        self.assertEqual(list_runs(Path(tempfile.gettempdir()) / 'khong_ton_tai_xyz'), [])


    def test_pick_run_by_target_and_data_file(self):
        with tempfile.TemporaryDirectory() as folder:
            runs = Path(folder)
            specs = {'20261101_kaggle': {}, '20261102_om': {'targets': ['tmax', 'tmin', 'rain'], 'data_file': 'open_meteo.csv'},
                     '20261103_om_weather': {'targets': ['weather'], 'data_file': 'open_meteo.csv'}}
            for name, meta in specs.items():
                (runs / name / 'models').mkdir(parents=True)
                (runs / name / 'models' / 'metadata.json').write_text(json.dumps(meta), encoding='utf-8')
            found = list_runs(runs)
            self.assertEqual(pick_run(found, 'weather', 'weather.csv').name, '20261101_kaggle')  # thiếu khoá = bản Kaggle
            self.assertEqual(pick_run(found, 'tmax', 'open_meteo.csv').name, '20261102_om')
            self.assertIsNone(pick_run(found, 'tmax', 'khac.csv'))
            self.assertEqual([r.name for r in runs_with(found, 'weather')], ['20261103_om_weather', '20261101_kaggle'])
            self.assertEqual([r.name for r in runs_with(found, 'rain')], ['20261102_om', '20261101_kaggle'])
            self.assertEqual(describe_run(runs / '20261101_kaggle', found), 'Kaggle')  # chỉ một bản Kaggle
            # hai bản cùng nguồn Open-Meteo: thêm tên lần chạy để phân biệt
            self.assertEqual(describe_run(runs / '20261102_om', found), 'Open-Meteo (20261102_om)')
            self.assertEqual(describe_run(runs / '20261102_om'), 'Open-Meteo')
            self.assertEqual(run_contents(runs / '20261101_kaggle'),
                             'loại thời tiết, nhiệt độ cao, nhiệt độ thấp, lượng mưa')


if __name__ == '__main__':
    unittest.main()
