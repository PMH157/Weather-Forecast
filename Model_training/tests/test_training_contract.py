"""Kiểm tra cấu hình và hợp đồng dữ liệu, không train classifier."""
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modeling import (baseline_predictions, build_pipeline, load_training_data,
                      ranked_candidates, validate_schema)
from train import main, make_candidates


def schema_example():
    return {'feature_columns':['city','temperature_max_c'], 'numeric_features':['temperature_max_c'],
            'categorical_features':['city'], 'label_to_code':{'sunny_or_few_clouds':0,'cloudy':1,'rain':2}}


def write_fixture(directory, overlap=False):
    schema = schema_example()
    (directory/'feature_schema.json').write_text(json.dumps(schema), encoding='utf-8')
    for split, start, labels in [('train','2020-01-01',[0,1,2]),
                                  ('validation','2020-01-03' if overlap else '2020-01-04',[2,1,0])]:
        dates = pd.date_range(start, periods=3)
        frame = pd.DataFrame({'city':['Nha Trang']*3, 'temperature_max_c':[25,26,27],
                              'sample_id':[split+str(i) for i in range(3)],
                              'origin_date':dates, 'target_date':dates+pd.Timedelta(days=1),
                              'target':labels, 'weather_today_code':[0,1,2], 'split':split})
        frame.to_csv(directory/(split+'.csv'), index=False)
    return schema


class TrainingContract(unittest.TestCase):
    def test_schema_rejects_target_as_feature(self):
        schema = schema_example()
        schema['feature_columns'].append('target')
        schema['numeric_features'].append('target')
        with self.assertRaises(ValueError):
            validate_schema(schema)

    def test_schema_rejects_duplicate_label_codes(self):
        schema = schema_example()
        schema['label_to_code']['rain'] = 1
        with self.assertRaises(ValueError):
            validate_schema(schema)

    def test_scaled_models_and_trees_have_the_right_steps(self):
        for name in ['logistic_regression','knn','decision_tree','random_forest','svm']:
            pipeline = build_pipeline(name, schema_example())
            numeric = pipeline.named_steps['preprocess'].transformers[0][1]
            self.assertEqual('scaler' in dict(numeric.steps), name in ['logistic_regression','knn','svm'])
            self.assertFalse(hasattr(pipeline.named_steps['classifier'], 'classes_'))

    def test_default_config_has_thirty_candidates_without_fitting(self):
        config = json.loads((ROOT/'config/training.json').read_text())
        with mock.patch.object(Pipeline,'fit',side_effect=AssertionError('No fit during configuration')):
            self.assertEqual(len(make_candidates(config,schema_example())),30)

    def test_quick_config_has_one_candidate_per_model(self):
        config = json.loads((ROOT/'config/training.json').read_text())
        self.assertEqual(len(make_candidates(config,schema_example(),quick=True)),5)

    def test_selection_prioritizes_macro_f1_over_accuracy(self):
        records = [{'status':'ok','candidate_id':'A','macro_f1':0.6,'accuracy':0.95},
                   {'status':'ok','candidate_id':'B','macro_f1':0.7,'accuracy':0.8},
                   {'status':'failed','candidate_id':'C','macro_f1':1,'accuracy':1}]
        self.assertEqual(ranked_candidates(records)[0]['candidate_id'],'B')

    def test_temporal_data_contract_accepts_available_train_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            write_fixture(Path(temp))
            train, validation, _ = load_training_data(temp)
            self.assertEqual(train.target_date.max(),validation.origin_date.min())

    def test_temporal_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            write_fixture(Path(temp),overlap=True)
            with self.assertRaises(ValueError):
                load_training_data(temp)

    def test_baseline_majority_uses_train_and_persistence_uses_today(self):
        train = pd.DataFrame({'target':[2,2,0]})
        validation = pd.DataFrame({'target':[0,0,0], 'weather_today_code':[0,np.nan,1]})
        result = baseline_predictions(train,validation,[0,1,2])
        np.testing.assert_array_equal(result['majority_train'],[2,2,2])
        np.testing.assert_array_equal(result['persistence_today'],[0,2,1])

    def test_check_data_never_calls_fit(self):
        with tempfile.TemporaryDirectory() as temp:
            write_fixture(Path(temp))
            with mock.patch.object(Pipeline,'fit',side_effect=AssertionError('No training permitted')) as fit:
                with contextlib.redirect_stdout(io.StringIO()):
                    status = main(['--data-dir',temp,'--check-data'])
            self.assertEqual(status,0)
            fit.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
