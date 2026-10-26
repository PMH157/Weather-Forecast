"""Hồi quy lỗi in tiếng Việt trên Windows; không huấn luyện classifier thật."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import modeling
import predict
import train as training
import training_visualization as visualization
from test_training_contract import write_fixture


@contextlib.contextmanager
def console_streams(encoding):
    buffers = [io.BytesIO(), io.BytesIO()]
    streams = [io.TextIOWrapper(b, encoding=encoding, errors='strict', write_through=True) for b in buffers]
    try:
        with contextlib.redirect_stdout(streams[0]), contextlib.redirect_stderr(streams[1]):
            yield streams, buffers
    finally:
        for stream in streams:
            stream.close()


class PipelineFixture:
    """Chỉ trả nhãn cố định để kiểm tra luồng lưu file và log; không học dữ liệu."""
    def __init__(self):
        self.named_steps = {'preprocess': mock.Mock()}
        self.named_steps['preprocess'].get_feature_names_out.return_value = np.array(['city_A', 'temperature'])

    def set_params(self, **params):
        return self

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class ConsoleCompatibility(unittest.TestCase):
    def test_cp1252_handles_stdout_stderr_and_unicode_paths(self):
        with console_streams('cp1252') as (streams, buffers):
            modeling.configure_console()
            print('Đã chọn mô hình: Hà Nội')
            print(Path('Kết quả') / 'dự đoán.csv', file=sys.stderr)
            self.assertEqual(streams[0].encoding, 'cp1252')
            self.assertEqual(streams[1].encoding, 'cp1252')
            self.assertIn(b'\\u0110', buffers[0].getvalue())
            self.assertIn('csv', buffers[1].getvalue().decode('cp1252'))

    def test_utf8_preserves_vietnamese_output_and_file_contents(self):
        message = 'Đã chọn mô hình: Nắng hoặc ít mây'
        with tempfile.TemporaryDirectory() as temporary, console_streams('utf-8') as (_, buffers):
            modeling.configure_console()
            print(message)
            print(message, file=sys.stderr)
            path = Path(temporary) / 'report.json'
            modeling.write_json(path, {'label': message})
            self.assertEqual(buffers[0].getvalue().decode('utf-8'), message + '\n')
            self.assertEqual(buffers[1].getvalue().decode('utf-8'), message + '\n')
            self.assertIn(message, path.read_text(encoding='utf-8'))

    def test_stringio_capture_remains_usable(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            modeling.configure_console()
            print('Đã chọn mô hình')
        self.assertEqual(output.getvalue(), 'Đã chọn mô hình\n')

    def test_all_cli_help_commands_work_with_strict_cp1252(self):
        environment = os.environ.copy()
        environment['PYTHONIOENCODING'] = 'cp1252:strict'
        environment.pop('PYTHONUTF8', None)
        for script in ['train.py', 'training_visualization.py', 'predict.py']:
            with self.subTest(script=script):
                result = subprocess.run([sys.executable, str(ROOT / script), '--help'],
                                        env=environment, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr.decode('cp1252'))
                self.assertIn('usage:', result.stdout.decode('cp1252'))
                self.assertNotIn(b'UnicodeEncodeError', result.stderr)

    def test_main_error_message_can_be_printed_on_cp1252(self):
        with console_streams('cp1252') as (_, buffers):
            with mock.patch.object(training, 'load_training_data', side_effect=ValueError('Dữ liệu thiếu nhãn')):
                status = training.main(['--check-data'])
            self.assertEqual(status, 2)
            self.assertIn('nh', buffers[1].getvalue().decode('cp1252'))

    def test_training_workflow_reaches_final_save_with_mock_estimator(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / 'data'
            data.mkdir()
            write_fixture(data)
            train, validation, schema = modeling.load_training_data(data)
            output = Path(temporary) / 'Đầu ra mô hình'
            config = {'random_state': 42, 'n_jobs': 1, 'visualization': {'enabled': False}}

            def fake_dump(value, path):
                Path(path).write_bytes(b'console regression fixture; not a trained model')

            with console_streams('cp1252') as (_, buffers):
                with mock.patch.object(training, 'build_pipeline', side_effect=lambda *a, **k: PipelineFixture()), \
                     mock.patch.object(training, 'clone', side_effect=lambda value: PipelineFixture()), \
                     mock.patch.object(training.joblib, 'dump', side_effect=fake_dump):
                    selection = training.train_models(train, validation, schema, config,
                        [('example_001', 'logistic_regression', {})], output, data)
                self.assertIn(b'\\u0110', buffers[0].getvalue())
            self.assertEqual(selection['selected_model'], 'logistic_regression')
            self.assertTrue((output / 'models/final_pipeline.joblib').is_file())
            self.assertFalse(selection['test_used'])

    def test_visualization_progress_and_failure_message_work_on_cp1252(self):
        with tempfile.TemporaryDirectory() as temporary, console_streams('cp1252') as (_, buffers):
            with mock.patch.object(visualization, 'learning_curve_rows', side_effect=ValueError('Dữ liệu thiếu')):
                manifest = visualization.create_training_visualizations(None, None, {}, {},
                    [{'model': 'knn', 'params': '{}'}], temporary)
            self.assertEqual(manifest['failed_curves'], 1)
            self.assertIn(b'knn_learning_curve', buffers[0].getvalue())
            index = pd.read_csv(Path(temporary) / 'results/visualizations/curves_index.csv')
            self.assertIn('Dữ liệu thiếu', index.error.iloc[0])

    def test_predict_unicode_output_path_works_without_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schema = {'feature_columns': ['city', 'temperature'], 'numeric_features': ['temperature'],
                      'label_to_code': {'sunny': 0}, 'label_display_vi': {'sunny': 'Nắng hoặc ít mây'}}
            modeling.write_json(root / 'feature_schema.json', schema)
            pd.DataFrame({'city': ['Hà Nội'], 'temperature': [28]}).to_csv(root / 'input.csv', index=False)
            output = root / 'Kết quả dự đoán.csv'
            argv = ['predict.py', '--model', str(root / 'final_pipeline.joblib'),
                    '--input', str(root / 'input.csv'), '--output', str(output)]
            with console_streams('cp1252'), mock.patch.object(sys, 'argv', argv):
                with mock.patch.object(predict.joblib, 'load', return_value=PipelineFixture()):
                    predict.main()
            frame = pd.read_csv(output)
            self.assertEqual(frame.label.iloc[0], 'Nắng hoặc ít mây')


if __name__ == '__main__':
    unittest.main(verbosity=2)
