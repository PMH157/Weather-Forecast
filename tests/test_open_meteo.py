"""Kiểm thử phần lấy số liệu Open-Meteo. Không gọi mạng: dùng phản hồi giả. Chạy: python -m unittest discover -s tests -v"""
from datetime import date, datetime, timezone
import io
import json
from pathlib import Path
import sys
import unittest
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features import CITIES, load_weather  # noqa: E402
from open_meteo import fetch_json, to_weather_frame, vietnam_yesterday, wind_label  # noqa: E402


def fake_block(days, base=0.0):
    n = len(days)
    return {'daily': {'time': days, 'temperature_2m_max': [30.0 + base] * n, 'temperature_2m_min': [22.0 + base] * n,
                      'precipitation_sum': [1.5] * n, 'wind_speed_10m_mean': [9.0] * n,
                      'wind_direction_10m_dominant': [359] * n, 'relative_humidity_2m_mean': [80.0] * n,
                      'cloud_cover_mean': [60.0] * n, 'pressure_msl_mean': [1010.0] * n}}


class WindLabelTest(unittest.TestCase):
    def test_wind_label_rounds_to_16_points(self):
        self.assertEqual([wind_label(d) for d in [0, 11, 12, 90, 337.5, 359, 360]],
                         ['N', 'N', 'NNE', 'E', 'NNW', 'N', 'N'])


class ConvertTest(unittest.TestCase):
    def test_to_weather_frame_matches_csv_schema_and_city_order(self):
        days = ['2026-11-06', '2026-11-07', '2026-11-08']
        frame = to_weather_frame([fake_block(days, 0), fake_block(days, 1), fake_block(days, 2)])
        self.assertEqual(list(frame.columns),
                         ['province', 'max', 'min', 'wind', 'wind_d', 'rain', 'humidi', 'cloud', 'pressure', 'date'])
        self.assertEqual(frame.groupby('province')['max'].first().to_dict(),
                         {'Ha Noi': 30.0, 'Ho Chi Minh City': 31.0, 'Nha Trang': 32.0})
        wide = load_weather(io.StringIO(frame.to_csv(index=False)))  # đi qua đúng đường kiểm tra của CSV
        self.assertEqual(list(wide['max'].columns), CITIES)
        self.assertEqual(wide['wind_deg'].iloc[0, 0], 0.0)

    def test_to_weather_frame_drops_days_from_until(self):
        days = ['2026-11-07', '2026-11-08', '2026-11-09']
        frame = to_weather_frame([fake_block(days)] * 3, until=date(2026, 11, 8))
        self.assertEqual(sorted(frame.date.unique()), ['2026-11-07', '2026-11-08'])

    def test_missing_values_stay_missing(self):
        block = fake_block(['2026-11-08'])
        block['daily']['cloud_cover_mean'] = [None]
        frame = to_weather_frame([block] * 3)
        self.assertTrue(frame.cloud.isna().all())


class FetchTest(unittest.TestCase):
    def test_fetch_retries_on_429_then_succeeds(self):
        calls, waits = [], []

        def opener(url, timeout):
            calls.append(url)
            if len(calls) < 3:
                raise urllib.error.HTTPError(url, 429, 'Too Many Requests', {}, None)
            return io.BytesIO(json.dumps([{'daily': {}}]).encode())

        self.assertEqual(fetch_json('https://x', opener=opener, sleep=waits.append), [{'daily': {}}])
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(waits), 2)

    def test_fetch_gives_vietnamese_error_after_retries(self):
        def opener(url, timeout):
            raise urllib.error.URLError('no network')
        with self.assertRaisesRegex(ValueError, 'Open-Meteo'):
            fetch_json('https://x', opener=opener, sleep=lambda s: None, retries=2)


    def test_rate_limit_message_says_wait_not_check_internet(self):
        def opener(url, timeout):
            raise urllib.error.HTTPError(url, 429, 'Too Many Requests', {}, None)
        with self.assertRaisesRegex(ValueError, 'giới hạn số lần gọi'):
            fetch_json('https://x', opener=opener, sleep=lambda s: None, retries=2)


class DateTest(unittest.TestCase):
    def test_vietnam_yesterday_uses_utc_plus_7(self):
        # 18:30 UTC ngày 08/11 đã là 01:30 ngày 09/11 ở Việt Nam, nên "hôm qua" là 08/11.
        self.assertEqual(vietnam_yesterday(datetime(2026, 11, 8, 18, 30, tzinfo=timezone.utc)), date(2026, 11, 8))
        self.assertEqual(vietnam_yesterday(datetime(2026, 11, 8, 10, 0, tzinfo=timezone.utc)), date(2026, 11, 7))


if __name__ == '__main__':
    unittest.main()
