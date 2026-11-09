"""Kiểm thử phần đọc dữ liệu, nhãn, đặc trưng và chia tập. Chạy: python -m unittest discover -s tests -v"""
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features import (CITIES, HORIZONS, assign_split, build_dataset, configure_console, load_weather, split_dates,  # noqa: E402
                      make_features, make_targets, weather_code)
import io  # noqa: E402

COLUMNS = ['province', 'max', 'min', 'wind', 'wind_d', 'rain', 'humidi', 'cloud', 'pressure', 'date']


def make_rows(dates, cities=CITIES, **overrides):
    """Tạo các dòng CSV giả: mỗi (thành phố, ngày) một dòng với giá trị mặc định."""
    rows = []
    for city in cities:
        for date in dates:
            row = dict(province=city, max=30, min=22, wind=10, wind_d='N', rain=0.0,
                       humidi=80, cloud=40, pressure=1010, date=date)
            row.update(overrides)
            rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


class CsvCase(unittest.TestCase):
    def write_csv(self, frame):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / 'weather.csv'
        frame.to_csv(path, index=False)
        return path


class LoadWeatherTest(CsvCase):
    def test_load_reindexes_calendar(self):
        wide = load_weather(self.write_csv(make_rows(['2020-01-01', '2020-01-03'])))
        self.assertEqual(len(wide['rain']), 3)
        self.assertTrue(wide['rain'].loc['2020-01-02'].isna().all())

    def test_load_drops_hanoi_duplicate(self):
        frame = pd.concat([make_rows(['2020-01-01']), make_rows(['2020-01-01'], cities=['Hanoi'])])
        wide = load_weather(self.write_csv(frame))
        self.assertEqual(list(wide['max'].columns), CITIES)

    def test_load_wind_degrees(self):
        frame = make_rows(['2020-01-01', '2020-01-02'])
        frame.loc[frame.date == '2020-01-01', 'wind_d'] = 'E'
        frame.loc[frame.date == '2020-01-02', 'wind_d'] = 'NNW'
        wide = load_weather(self.write_csv(frame))
        self.assertEqual(wide['wind_deg'].loc['2020-01-01', 'Ha Noi'], 90.0)
        self.assertEqual(wide['wind_deg'].loc['2020-01-02', 'Ha Noi'], 337.5)

    def test_load_missing_column_raises(self):
        path = self.write_csv(make_rows(['2020-01-01']).drop(columns='pressure'))
        with self.assertRaisesRegex(ValueError, 'pressure'):
            load_weather(path)

    def test_load_missing_city_raises(self):
        path = self.write_csv(make_rows(['2020-01-01'], cities=['Ha Noi', 'Ho Chi Minh City']))
        with self.assertRaisesRegex(ValueError, 'Nha Trang'):
            load_weather(path)

    def test_load_accepts_uploaded_file_object(self):
        """Trang web nhận file tải lên dưới dạng đối tượng file, không phải đường dẫn."""
        buffer = io.StringIO(make_rows(['2020-01-01', '2020-01-02']).to_csv(index=False))
        self.assertEqual(len(load_weather(buffer)['max']), 2)

    def test_load_duplicate_rows_raises(self):
        frame = pd.concat([make_rows(['2020-01-01']), make_rows(['2020-01-01'], cities=['Ha Noi'])])
        with self.assertRaises(ValueError):
            load_weather(self.write_csv(frame))


def wide_from_days(dates, value_by_date=None, cities=CITIES):
    """Bảng giả ngày x thành phố cho mọi biến; value_by_date ghi đè từng biến theo ngày."""
    rows = make_rows(dates, cities=cities)
    folder = tempfile.TemporaryDirectory()
    path = Path(folder.name) / 'w.csv'
    rows.to_csv(path, index=False)
    wide = load_weather(path)
    folder.cleanup()
    for (var, date), val in (value_by_date or {}).items():
        wide[var].loc[date] = val
    return wide


DAYS = pd.date_range('2020-01-01', '2020-01-20').strftime('%Y-%m-%d').tolist()


class LabelTest(unittest.TestCase):
    def test_weather_code_boundaries(self):
        rain = pd.DataFrame({'a': [1.0, 0.9, 0.9, np.nan]})
        cloud = pd.DataFrame({'a': [0.0, 50.0, 49.9, 30.0]})
        code = weather_code(rain, cloud)['a'].tolist()
        self.assertEqual(code[:3], [2.0, 1.0, 0.0])
        self.assertTrue(math.isnan(code[3]))

    def test_targets_use_calendar_day(self):
        days = [d for d in DAYS if d != '2020-01-06']
        targets = make_targets(wide_from_days(days), 1)
        self.assertTrue(math.isnan(targets.loc[(pd.Timestamp('2020-01-05'), 'Ha Noi'), 'weather_h1']))
        self.assertEqual(targets.loc[(pd.Timestamp('2020-01-04'), 'Ha Noi'), 'weather_h1'], 0.0)

    def test_targets_values_of_day_t_plus_h(self):
        wide = wide_from_days(DAYS, {('max', '2020-01-08'): 35.0, ('rain', '2020-01-08'): 5.0})
        row = make_targets(wide, 3).loc[(pd.Timestamp('2020-01-05'), 'Nha Trang')]
        self.assertEqual((row['tmax_h3'], row['rain_h3'], row['weather_h3']), (35.0, 5.0, 2.0))


class FeatureTest(unittest.TestCase):
    def test_features_no_future_leak(self):
        t = pd.Timestamp('2020-01-10')
        base = make_features(wide_from_days(DAYS), 1)
        changed = wide_from_days(DAYS)
        for var in changed:
            changed[var].loc[t + pd.Timedelta(days=1):] = changed[var].loc[t + pd.Timedelta(days=1):] + 7
        pd.testing.assert_frame_equal(base.loc[t], make_features(changed, 1).loc[t])

    def test_lag_is_nan_when_previous_day_missing(self):
        days = [d for d in DAYS if d != '2020-01-09']
        feats = make_features(wide_from_days(days), 1)
        self.assertTrue(math.isnan(feats.loc[(pd.Timestamp('2020-01-10'), 'Ha Noi'), 'pressure_lag1']))
        self.assertEqual(feats.loc[(pd.Timestamp('2020-01-10'), 'Ha Noi'), 'pressure_lag2'], 1010.0)

    def test_season_uses_target_date(self):
        wide = wide_from_days(pd.date_range('2020-12-20', '2021-01-05').strftime('%Y-%m-%d').tolist())
        feats = make_features(wide, 3)
        self.assertAlmostEqual(feats.loc[(pd.Timestamp('2020-12-29'), 'Ha Noi'), 'doy_cos'],
                               math.cos(2 * math.pi * 1 / 365.25))

    def test_features_have_city_column_and_no_missing_origin_days(self):
        days = [d for d in DAYS if d != '2020-01-09']
        feats = make_features(wide_from_days(days), 1)
        self.assertNotIn(pd.Timestamp('2020-01-09'), feats.index.get_level_values('origin_date'))
        self.assertEqual(feats.loc[(pd.Timestamp('2020-01-10'), 'Nha Trang'), 'city'], 'Nha Trang')


class SplitTest(unittest.TestCase):
    def test_split_boundaries(self):
        # 65% Train / 15% Validation / 20% Test, đệm 3 ngày ở cuối Train và cuối Validation.
        dates = pd.Series(pd.to_datetime(['2017-02-02', '2017-02-03', '2017-02-05', '2017-02-06',
                                          '2018-12-17', '2018-12-18', '2018-12-20', '2018-12-21']))
        self.assertEqual(assign_split(dates, '2017-02-05', '2018-12-20').tolist(),
                         ['train', 'gap', 'gap', 'validation', 'validation', 'gap', 'gap', 'test'])

    def test_split_dates_use_65_and_80_percent_of_days(self):
        wide = wide_from_days(pd.date_range('2020-01-01', periods=100).strftime('%Y-%m-%d').tolist())
        self.assertEqual(split_dates(wide), (pd.Timestamp('2020-03-05'), pd.Timestamp('2020-03-20')))

    def test_split_dates_reproduce_kaggle_cutoffs(self):
        wide = load_weather(Path(__file__).resolve().parent.parent / 'data' / 'weather.csv')
        self.assertEqual(split_dates(wide), (pd.Timestamp('2017-02-05'), pd.Timestamp('2018-12-20')))

    def test_split_no_target_crosses_boundary(self):
        wide = wide_from_days(pd.date_range('2017-01-01', '2019-01-31').strftime('%Y-%m-%d').tolist())
        for h in HORIZONS:
            X, _, split = build_dataset(wide, h)
            day = X.index.get_level_values('origin_date')
            first = {name: day[split == name].min() for name in ['validation', 'test']}
            last = {name: day[split == name].max() for name in ['train', 'validation']}
            self.assertLess(last['train'] + pd.Timedelta(days=h), first['validation'])
            self.assertLess(last['validation'] + pd.Timedelta(days=h), first['test'])

    def test_build_dataset_all_cities_same_day_same_split(self):
        wide = wide_from_days(pd.date_range('2017-01-20', '2017-02-20').strftime('%Y-%m-%d').tolist())
        X, targets, split = build_dataset(wide, 2)
        per_day = split.groupby(level='origin_date').nunique()
        self.assertTrue((per_day == 1).all())
        self.assertNotIn('gap', set(split))
        self.assertTrue(X.index.equals(targets.index) and X.index.equals(split.index))
        self.assertFalse(targets.isna().any().any())


class ConsoleTest(unittest.TestCase):
    def test_configure_console_tolerates_streams_without_reconfigure(self):
        """Spyder, Jupyter, IDLE thay stdout bằng đối tượng không có reconfigure: không được dừng chương trình."""
        configure_console([io.StringIO()])


if __name__ == '__main__':
    unittest.main()
