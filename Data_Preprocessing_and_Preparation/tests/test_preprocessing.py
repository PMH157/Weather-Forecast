"""Kiểm tra các tình huống dễ gây sai nhãn hoặc rò rỉ dữ liệu thời gian."""
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing import (DEFAULT_CONFIG, FEATURE_COLUMNS, NUMERIC_FEATURES, LABELS,
                           clean_daily, build_supervised, split_by_time, make_preprocessor,
                           make_weather_labels)


def fixture(drop_leap=False):
    rows = []
    for city_index, city in enumerate(['Ha Noi', 'Ho Chi Minh City', 'Nha Trang']):
        for i, date in enumerate(pd.date_range('2020-02-25', '2020-03-10')):
            if drop_leap and date == pd.Timestamp('2020-02-29'):
                continue
            rows.append({'province': city, 'max': 25 + city_index + (i % 4), 'min': 20 + city_index,
                         'wind': 36, 'wind_d': 'NW', 'rain': 2.0 if i % 3 == 2 else 0.0,
                         'humidi': 70 + i % 10, 'cloud': 60 if i % 3 == 1 else 20,
                         'pressure': 1010, 'date': date.strftime('%Y-%m-%d')})
    return pd.DataFrame(rows)


class PreprocessingSemantics(unittest.TestCase):
    def test_label_boundaries_and_unknowns(self):
        rain = pd.Series([1.0, 0.999, 0.999, np.nan, -1.0, 2.0, 0.0])
        cloud = pd.Series([0.0, 50.0, 49.999, 90.0, 10.0, np.nan, np.nan])
        labels = make_weather_labels(rain, cloud)
        self.assertEqual(labels.iloc[:3].tolist(), ['rain', 'cloudy', 'sunny_or_few_clouds'])
        self.assertTrue(labels.iloc[3:5].isna().all())
        self.assertEqual(labels.iloc[5], 'rain')
        self.assertTrue(pd.isna(labels.iloc[6]))

    def test_leap_gap_cannot_be_used_as_next_day(self):
        daily, _, _, _ = clean_daily(fixture(drop_leap=True))
        data, rejected = build_supervised(daily)
        before_gap = rejected[rejected.origin_date.eq(pd.Timestamp('2020-02-28'))]
        self.assertEqual(len(before_gap), 3)
        self.assertTrue(before_gap.target_date.eq(pd.Timestamp('2020-02-29')).all())
        march1 = data[data.origin_date.eq(pd.Timestamp('2020-03-01'))]
        self.assertTrue(march1.rain_mm_lag1.isna().all())
        self.assertTrue(march1.rain_mm_lag2.notna().all())
        self.assertTrue(march1.rain_mm_rolling3_mean.isna().all())

    def test_future_measurements_do_not_change_earlier_features(self):
        raw = fixture()
        first, _ = build_supervised(clean_daily(raw)[0])
        changed = raw.copy()
        changed.loc[changed.date.ge('2020-03-05'), ['max', 'min', 'rain', 'humidi', 'cloud']] = [44, 25, 300, 99, 100]
        second, _ = build_supervised(clean_daily(changed)[0])
        cutoff = pd.Timestamp('2020-03-04')
        a = first[first.origin_date.le(cutoff)][['sample_id'] + FEATURE_COLUMNS].reset_index(drop=True)
        b = second[second.origin_date.le(cutoff)][['sample_id'] + FEATURE_COLUMNS].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)

    def test_lags_stay_within_city(self):
        raw = fixture()
        raw.loc[raw.province.eq('Nha Trang'), 'rain'] = 100.0
        raw.loc[raw.province.eq('Ha Noi'), 'rain'] = 0.0
        data, _ = build_supervised(clean_daily(raw)[0])
        self.assertTrue(data.loc[data.city.eq('Nha Trang'), 'rain_mm_lag1'].dropna().eq(100).all())
        self.assertTrue(data.loc[data.city.eq('Ha Noi'), 'rain_mm_lag1'].dropna().eq(0).all())

    def test_missing_target_is_not_filled_with_a_class(self):
        raw = fixture()
        raw.loc[raw.date.eq('2020-03-03'), 'rain'] = np.nan
        data, rejected = build_supervised(clean_daily(raw)[0])
        self.assertEqual(len(rejected[rejected.target_date.eq(pd.Timestamp('2020-03-03'))]), 3)
        self.assertFalse(data.target_date.eq(pd.Timestamp('2020-03-03')).any())
        self.assertFalse(data.target.isna().any())

    def test_units_and_pressure_exclusion(self):
        daily, _, _, _ = clean_daily(fixture())
        self.assertTrue(np.allclose(daily.wind_speed_m_s, 10.0))
        self.assertTrue(np.allclose(daily.wind_direction_deg, 315.0))
        self.assertTrue(daily.pressure_raw.eq(1010.0).all())
        self.assertNotIn('pressure_raw', FEATURE_COLUMNS)
        self.assertNotIn('target', FEATURE_COLUMNS)

    def test_invalid_cells_preserve_other_measurements_and_extreme_rain(self):
        raw = fixture()
        raw.loc[0, ['max', 'min', 'humidi', 'cloud', 'rain', 'wind']] = [10, 20, 101, -1, -1, -5]
        raw.loc[1, 'rain'] = 596.4
        daily, audit, _, _ = clean_daily(raw)
        row = daily[(daily.city == 'Ha Noi') & daily.date.eq(pd.Timestamp('2020-02-25'))].iloc[0]
        self.assertTrue(pd.isna(row.temperature_max_c))
        self.assertTrue(pd.isna(row.temperature_min_c))
        self.assertTrue(pd.isna(row.humidity_pct))
        self.assertEqual(row.pressure_raw, 1010)
        self.assertEqual(audit['selected_observed_rows'], len(raw))
        self.assertEqual(daily.loc[(daily.city == 'Ha Noi') & daily.date.eq(pd.Timestamp('2020-02-26')), 'rain_mm'].iloc[0], 596.4)

    def test_preprocessing_uses_train_statistics_only(self):
        train = pd.DataFrame({col: [10.0, 12.0, 14.0] for col in NUMERIC_FEATURES})
        train['city'] = ['Ha Noi', 'Nha Trang', 'Ho Chi Minh City']
        train.loc[1, 'temperature_max_c'] = np.nan
        pre = make_preprocessor(scale=True).fit(train[FEATURE_COLUMNS])
        numeric = pre.named_transformers_['numeric']
        means = numeric.named_steps['scaler'].mean_.copy()
        medians = numeric.named_steps['imputer'].statistics_.copy()
        self.assertEqual(medians[0], 12.0)
        self.assertEqual(means[0], 12.0)
        validation = train.copy()
        validation.loc[:, NUMERIC_FEATURES] = 100000.0
        result = pre.transform(validation[FEATURE_COLUMNS])
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(numeric.named_steps['scaler'].mean_, means)
        np.testing.assert_array_equal(numeric.named_steps['imputer'].statistics_, medians)
        self.assertEqual(numeric.named_steps['scaler'].n_samples_seen_, len(train))

    def test_temporal_splits_keep_city_dates_together(self):
        data, _ = build_supervised(clean_daily(fixture())[0])
        combined, splits, _ = split_by_time(data)
        self.assertTrue(combined.groupby('target_date').split.nunique().eq(1).all())
        self.assertLessEqual(splits['train'].target_date.max(), splits['validation'].origin_date.min())
        self.assertLessEqual(splits['validation'].target_date.max(), splits['test'].origin_date.min())
        self.assertTrue(combined.target_date.sub(combined.origin_date).eq(pd.Timedelta(days=1)).all())

    def test_exact_duplicates_removed_but_conflicting_duplicates_fail(self):
        raw = fixture()
        _, audit, _, _ = clean_daily(pd.concat([raw, raw.iloc[[0]]], ignore_index=True))
        self.assertEqual(audit['exact_duplicate_rows_removed'], 1)
        conflict = raw.iloc[[0]].copy()
        conflict['wind'] = 40
        with self.assertRaisesRegex(ValueError, 'Conflicting duplicate'):
            clean_daily(pd.concat([raw, conflict], ignore_index=True))

    def test_hanoi_alias_policy_is_consistent_for_entire_series(self):
        raw = fixture()
        alias = raw[raw.province.eq('Ha Noi')].copy()
        alias['province'] = 'Hanoi'
        alias['rain'] = 200.0
        daily, audit, _, _ = clean_daily(pd.concat([alias, raw], ignore_index=True))
        self.assertEqual(audit['hanoi_alternate_series_rows_excluded'], len(alias))
        self.assertLess(daily.loc[daily.city.eq('Ha Noi'), 'rain_mm'].max(), 200)


if __name__ == '__main__':
    unittest.main(verbosity=2)
