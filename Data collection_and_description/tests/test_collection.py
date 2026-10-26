"""Kiểm thử thu thập API bằng phản hồi giả lập, không gọi mạng thật."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collect_openweather import collect, flatten, load_api_key


class CollectionBehavior(unittest.TestCase):
    def test_environment_takes_priority_over_local_file(self):
        with tempfile.TemporaryDirectory() as temp:
            env = Path(temp) / '.env'
            env.write_text('OPENWEATHER_API_KEY=local-example-value\n', encoding='utf-8')
            with mock.patch.dict(os.environ, {'OPENWEATHER_API_KEY': 'environment-example-value'}, clear=True):
                self.assertEqual(load_api_key(env), 'environment-example-value')

    def test_local_env_supports_comments_quotes_and_export(self):
        with tempfile.TemporaryDirectory() as temp:
            env = Path(temp) / '.env'
            env.write_text('# Comment\nOTHER=unused\nexport OPENWEATHER_API_KEY="local-example-value"\n', encoding='utf-8-sig')
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(load_api_key(env), 'local-example-value')

    def test_placeholder_is_not_a_real_key(self):
        with tempfile.TemporaryDirectory() as temp:
            env = Path(temp) / '.env'
            env.write_text('OPENWEATHER_API_KEY=YOUR_KEY_HERE\n', encoding='utf-8')
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(load_api_key(env), '')

    def test_snapshot_preserves_missing_rain_and_utc_offset(self):
        payload = {'main': {'temp': 28}, 'wind': {'speed': 2, 'deg': 90}, 'dt': 0}
        city = {'city': 'Nha Trang', 'lat': 12.2388, 'lon': 109.1967}
        result = flatten(payload, city, '1970-01-01T00:00:01+00:00')
        self.assertIsNone(result['rain_1h_mm_per_h'])
        self.assertFalse(result['rain_1h_field_present'])
        self.assertEqual(result['observed_at_local'], '1970-01-01T07:00:00+07:00')
        self.assertEqual(result['time_resolution'], 'instant_snapshot')

    def test_http_error_does_not_print_request_key(self):
        dummy_key = 'dummy-test-key-never-valid'
        error = urllib.error.HTTPError('https://example.invalid/?appid=' + dummy_key, 401,
                                        'unauthorized', None, None)
        with tempfile.TemporaryDirectory() as temp:
            output = io.StringIO()
            with mock.patch('collect_openweather.urllib.request.urlopen', side_effect=error):
                with contextlib.redirect_stdout(output):
                    success = collect(Path(temp), dummy_key)
            self.assertFalse(success)
            self.assertNotIn(dummy_key, output.getvalue())
            log = json.loads(next(Path(temp).glob('collection_log_*.json')).read_text())
            self.assertEqual(log['success_count'], 0)
            self.assertEqual(len(log['cities']), 3)
            self.assertTrue(all(row['http_status'] == 401 for row in log['cities']))

    def test_malformed_api_fields_return_failure_without_crashing(self):
        payloads = [[], {'cod': 200, 'main': {}, 'dt': 0, 'wind': None}]
        for payload in payloads:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as temp:
                def response(*args, **kwargs):
                    return io.BytesIO(json.dumps(payload).encode('utf-8'))
                with mock.patch('collect_openweather.urllib.request.urlopen', side_effect=response):
                    with contextlib.redirect_stdout(io.StringIO()):
                        success = collect(Path(temp), 'dummy-test-value')
                self.assertFalse(success)
                log = json.loads(next(Path(temp).glob('collection_log_*.json')).read_text())
                self.assertEqual(log['success_count'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
