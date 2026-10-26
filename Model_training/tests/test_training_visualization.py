"""Kiểm tra trục thời gian, số đo và tích hợp bằng mock; không train classifier."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import training_visualization as visual


def fixture():
    train = pd.DataFrame([
        {'target_date': day, 'city': city, 'temperature_max_c': 20 + day.day,
         'target': code}
        for day in pd.date_range('2020-01-01', periods=4)
        for code, city in enumerate(['A', 'B', 'C'])
    ])
    validation = train.iloc[-3:].copy()
    validation['target_date'] += pd.Timedelta(days=10)
    schema = {'feature_columns': ['city', 'temperature_max_c'],
              'numeric_features': ['temperature_max_c'], 'categorical_features': ['city'],
              'label_to_code': {'sunny_or_few_clouds': 0, 'cloudy': 1, 'rain': 2}}
    return train, validation, schema, {'random_state': 42, 'n_jobs': 1}


class ProbabilitiesFixture:
    classes_ = np.array([2, 0, 1])

    def predict(self, X):
        return X['expected'].to_numpy()

    def predict_proba(self, X):
        values = np.full((len(X), 3), 0.1)
        for i, code in enumerate(X['expected']):
            values[i, list(self.classes_).index(code)] = 0.8
        return values


class PredictionFixture:
    def predict(self, X):
        return X['prediction'].to_numpy()


def metric_fixture():
    return {'train_accuracy': 0.8, 'validation_accuracy': 0.7,
            'train_macro_f1': 0.75, 'validation_macro_f1': 0.65,
            'loss_name': 'log_loss', 'train_loss': 0.4, 'validation_loss': 0.6}


class TrainingVisualization(unittest.TestCase):
    def test_settings_add_full_prefix_without_mutating_defaults(self):
        original = deepcopy(visual.DEFAULT_VISUALIZATION)
        settings = visual.visualization_settings({'learning_fractions': [0.5, 0.1, 0.5]})
        self.assertEqual(settings['learning_fractions'], [0.1, 0.5, 1.0])
        self.assertEqual(visual.DEFAULT_VISUALIZATION, original)

    def test_invalid_settings_are_rejected(self):
        for value in [[], {'enabled': 1}, {'learning_fractions': [0]},
                      {'learning_fractions': [float('nan')]}, {'learning_fractions': None},
                      {'logistic_iteration_checkpoints': [True]}, {'dpi': 0}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                visual.visualization_settings(value)

    def test_prefixes_preserve_whole_days_and_full_train(self):
        train, _, _, _ = fixture()
        prefixes = visual.chronological_prefixes(train.iloc[::-1], [0.25, 0.26, 0.5], [0, 1, 2])
        self.assertEqual([len(p[2]) for p in prefixes], [3, 6, 12])
        for _, cutoff, prefix in prefixes:
            self.assertTrue(prefix.target_date.le(cutoff).all())
            self.assertTrue(prefix.groupby('target_date').size().eq(3).all())
        self.assertEqual(set(prefixes[-1][2].index), set(train.index))

    def test_prefixes_skip_incomplete_classes_or_too_few_neighbors(self):
        train, _, _, _ = fixture()
        train.loc[train.target_date.eq(train.target_date.min()), 'target'] = 0
        prefixes = visual.chronological_prefixes(train, [0.25, 0.5], [0, 1, 2], minimum_rows=7)
        self.assertEqual([len(p[2]) for p in prefixes], [12])

    def test_log_loss_respects_estimator_class_column_order(self):
        X = pd.DataFrame({'expected': [0, 1, 2]})
        result = visual.curve_metrics(ProbabilitiesFixture(), X, [0, 1, 2], X, [0, 1, 2], [0, 1, 2])
        self.assertEqual(result['train_accuracy'], 1.0)
        self.assertAlmostEqual(result['validation_loss'], -np.log(0.8))
        self.assertEqual(result['loss_name'], 'log_loss')

    def test_no_probability_model_uses_error_instead_of_log_loss(self):
        X = pd.DataFrame({'prediction': [0, 0, 2]})
        result = visual.curve_metrics(PredictionFixture(), X, [0, 1, 2], X, [0, 1, 2], [0, 1, 2])
        self.assertEqual(result['loss_name'], 'classification_error')
        self.assertAlmostEqual(result['validation_loss'], 1 / 3)

    def test_learning_fits_only_prefix_and_keeps_validation_fixed(self):
        train, validation, schema, config = fixture()
        fitted = []
        measured = []

        def fake_fit(self, X, y):
            fitted.append(X.copy())
            return self

        def fake_metrics(pipeline, X_train, y_train, X_val, y_val, codes):
            measured.append((len(X_train), X_val.copy()))
            return metric_fixture()

        with mock.patch.object(Pipeline, 'fit', fake_fit), mock.patch.object(visual, 'curve_metrics', fake_metrics):
            rows = visual.learning_curve_rows('knn', {'classifier__n_neighbors': 5},
                                             train, validation, schema, config,
                                             visual.visualization_settings({'learning_fractions': [0.25, 0.5]}))
        self.assertEqual(rows.n_train_rows.tolist(), [6, 12])
        self.assertEqual([len(x) for x in fitted], [6, 12])
        for _, actual_validation in measured:
            pd.testing.assert_frame_equal(actual_validation, validation[schema['feature_columns']])

    def test_logistic_records_actual_iterations_and_stops_on_convergence(self):
        train, validation, schema, config = fixture()
        budgets = []

        def fake_fit(self, X, y):
            classifier = self.named_steps['classifier']
            budgets.append(classifier.max_iter)
            classifier.n_iter_ = np.array([min(classifier.max_iter, 4)])
            if classifier.max_iter < 4:
                warnings.warn('fixture has not converged', ConvergenceWarning)
            return self

        with mock.patch.object(Pipeline, 'fit', fake_fit), mock.patch.object(visual, 'curve_metrics', return_value=metric_fixture()):
            rows = visual.logistic_iteration_rows({}, train, validation, schema, config,
                                                 visual.visualization_settings({'logistic_iteration_checkpoints': [1, 2, 5, 10]}))
        self.assertEqual(budgets, [1, 2, 5])
        self.assertEqual(rows.actual_iterations.tolist(), [1, 2, 4])
        self.assertEqual(rows.converged.tolist(), [False, False, True])

    def test_forest_grows_same_estimator_and_preprocesses_once(self):
        train, validation, schema, config = fixture()
        preprocess = mock.Mock()
        transformed = np.zeros((len(train), 2))
        preprocess.fit_transform.return_value = transformed
        classifier = mock.Mock(n_estimators=6)

        def fake_set_params(**params):
            for key, value in params.items():
                setattr(classifier, key, value)
            return classifier

        def fake_fit(X, y):
            self.assertIs(X, transformed)
            self.assertTrue(classifier.warm_start)
            classifier.estimators_ = [object()] * classifier.n_estimators
            return classifier

        classifier.set_params.side_effect = fake_set_params
        classifier.fit.side_effect = fake_fit
        pipeline = mock.Mock(named_steps={'preprocess': preprocess, 'classifier': classifier})
        with mock.patch.object(visual, 'build_pipeline', return_value=pipeline), mock.patch.object(visual, 'curve_metrics', return_value=metric_fixture()):
            rows = visual.random_forest_tree_rows({}, train, validation, schema, config,
                                                 visual.visualization_settings({'random_forest_tree_checkpoints': [2, 4, 8]}))
        self.assertEqual(rows.n_trees.tolist(), [2, 4, 6])
        self.assertEqual(classifier.fit.call_count, 3)
        preprocess.fit_transform.assert_called_once()
        self.assertTrue(rows.fit_seconds.is_monotonic_increasing)

    def test_disabled_plots_perform_no_fits(self):
        train, validation, schema, config = fixture()
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(Pipeline, 'fit', side_effect=AssertionError('No fit')) as fit:
            result = visual.create_training_visualizations(train, validation, schema, config,
                        [{'model': 'logistic_regression', 'params': '{}'}], temporary, {'enabled': False})
            self.assertFalse(result['enabled'])
            self.assertEqual(result['successful_curves'], 0)
            fit.assert_not_called()

    def test_curve_failure_is_recorded_without_touching_final_model(self):
        train, validation, schema, config = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / 'models/final_pipeline.joblib'
            model.parent.mkdir()
            model.write_bytes(b'fixture: not a trained estimator')
            with mock.patch.object(visual, 'learning_curve_rows', side_effect=ValueError('fixture failure')):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = visual.create_training_visualizations(train, validation, schema, config,
                                [{'model': 'knn', 'params': '{}'}], temporary)
            self.assertEqual(result['failed_curves'], 1)
            self.assertEqual(model.read_bytes(), b'fixture: not a trained estimator')
            errors = pd.read_csv(Path(temporary) / 'results/visualizations/curves_index.csv')
            self.assertIn('fixture failure', errors.error.iloc[0])

    def test_standalone_refuses_changed_data_before_fitting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'models').mkdir()
            (root / 'models/selection.json').write_text(json.dumps({'source_sha256': {'train.csv': 'original-hash'}}))
            (root / 'train.csv').write_text('changed')
            with mock.patch.object(Pipeline, 'fit', side_effect=AssertionError('No fit')) as fit:
                with contextlib.redirect_stderr(io.StringIO()):
                    status = visual.main(['--run-dir', temporary, '--data-dir', temporary])
            self.assertEqual(status, 2)
            fit.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
