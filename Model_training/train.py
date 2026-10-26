"""Thành viên 3: người dùng tự chạy để huấn luyện, so sánh và chọn mô hình."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import ParameterGrid

from modeling import (MODEL_NAMES, baseline_predictions, build_pipeline, configure_console, load_training_data,
                      ranked_candidates, read_json, score_predictions, write_json)

ROOT = Path(__file__).resolve().parent


def make_candidates(config, schema, names=None, quick=False):
    selected = list(config['models']) if names is None else names
    if not selected or len(selected) != len(set(selected)):
        raise ValueError('Danh sách mô hình trống hoặc bị trùng')
    unknown = set(selected) - set(MODEL_NAMES)
    if unknown or set(selected) - set(config['models']):
        raise ValueError('Mô hình không có trong cấu hình: ' + ', '.join(sorted(unknown or set(selected)-set(config['models']))))
    candidates = []
    for model_name in selected:
        base = build_pipeline(model_name, schema, config['random_state'], config['n_jobs'])
        grid = list(ParameterGrid(config['models'][model_name]))
        if quick:
            grid = grid[:1]
        for number, params in enumerate(grid, start=1):
            # Chỉ kiểm tra tên tham số, chưa fit mô hình.
            clone(base).set_params(**params)
            candidates.append((f'{model_name}_{number:03d}', model_name, params))
    return candidates


def train_models(train, validation, schema, config, candidates, output, data_dir):
    configure_console()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Thư mục kết quả đã có dữ liệu; chọn --output-dir khác')
    results, models = output / 'results', output / 'models'
    results.mkdir(parents=True, exist_ok=True)
    models.mkdir(exist_ok=True)
    features = schema['feature_columns']
    codes = sorted(schema['label_to_code'].values())
    code_to_label = {code: label for label, code in schema['label_to_code'].items()}
    labels = [schema.get('label_display_vi', {}).get(code_to_label[code], code_to_label[code]) for code in codes]
    X_train, y_train = train[features], train.target.to_numpy()
    X_val, y_val = validation[features], validation.target.to_numpy()
    base_rows = []
    for name, prediction in baseline_predictions(train, validation, codes).items():
        base_rows.append({'baseline': name, **score_predictions(y_val, prediction, codes)})
    pd.DataFrame(base_rows).to_csv(results / 'baselines_validation.csv', index=False, encoding='utf-8-sig')
    records, errors = [], []
    best_pipeline, best_prediction, best_record = None, None, None
    write_json(results / 'training_config.json', config)
    for index, (candidate_id, model_name, params) in enumerate(candidates, start=1):
        print(f'[{index}/{len(candidates)}] {candidate_id}', flush=True)
        pipeline = build_pipeline(model_name, schema, config['random_state'], config['n_jobs'])
        pipeline.set_params(**params)
        start = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                pipeline.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            prediction = pipeline.predict(X_val)
            predict_seconds = time.perf_counter() - start
            row = {'candidate_id': candidate_id, 'model': model_name, 'status': 'ok',
                   **score_predictions(y_val, prediction, codes),
                   'fit_seconds': fit_seconds, 'predict_seconds': predict_seconds,
                   'params': json.dumps(params, ensure_ascii=False, sort_keys=True),
                   'warnings': ' | '.join(str(w.message) for w in captured)}
            records.append(row)
            if ranked_candidates(records)[0]['candidate_id'] == candidate_id:
                best_pipeline, best_prediction, best_record = pipeline, prediction.copy(), row.copy()
        except Exception as error:
            row = {'candidate_id': candidate_id, 'model': model_name, 'status': 'failed',
                   'params': json.dumps(params, ensure_ascii=False, sort_keys=True),
                   'error_type': type(error).__name__, 'error': str(error)}
            records.append(row)
            errors.append(row)
            print(f'  Thất bại: {type(error).__name__}: {error}', flush=True)
        # Ghi tiến trình để người chạy xem lại nếu dừng giữa chừng.
        pd.DataFrame(records).to_csv(results / 'all_candidates_validation.csv', index=False, encoding='utf-8-sig')
        write_json(results / 'search_errors.json', errors)
    ranking = ranked_candidates(records)
    if not ranking:
        raise RuntimeError('Không có cấu hình train thành công; xem results/search_errors.json')
    pd.DataFrame(ranking).to_csv(results / 'validation_ranking.csv', index=False, encoding='utf-8-sig')
    best_each = pd.DataFrame(ranking).drop_duplicates('model', keep='first')
    best_each.to_csv(results / 'best_by_model.csv', index=False, encoding='utf-8-sig')
    # Những dự đoán dưới đây được tạo khi mô hình chỉ fit trên Train.
    predictions = validation[['sample_id','city','origin_date','target_date','target']].copy()
    predictions['prediction'] = best_prediction
    predictions.to_csv(results / 'winner_validation_predictions.csv', index=False, encoding='utf-8-sig', date_format='%Y-%m-%d')
    matrix = confusion_matrix(y_val, best_prediction, labels=codes)
    pd.DataFrame(matrix, index=labels, columns=labels).rename_axis('actual').to_csv(
        results / 'winner_validation_confusion_matrix.csv', encoding='utf-8-sig')
    report = classification_report(y_val, best_prediction, labels=codes, target_names=labels,
                                   output_dict=True, zero_division=0)
    write_json(results / 'winner_validation_classification_report.json', report)
    print('Đã chọn bằng Validation:', best_record['model'], flush=True)
    print('Fit lại Pipeline đã chọn trên Train + Validation...', flush=True)
    final_pipeline = clone(best_pipeline)
    combined = pd.concat([train, validation], ignore_index=True)
    final_pipeline.fit(combined[features], combined.target.to_numpy())
    joblib.dump(final_pipeline, models / 'final_pipeline.joblib')
    write_json(models / 'feature_schema.json', schema)
    selection = {
        'selected_model': best_record['model'],
        'selected_candidate_id': best_record['candidate_id'],
        'selected_params': json.loads(best_record['params']),
        'selection_rule': 'Highest Validation Macro F1; ties use Validation Accuracy, then candidate_id',
        'validation_metrics': {name: best_record[name] for name in ['macro_f1','accuracy','macro_precision','macro_recall']},
        'validation_model_fitted_on': 'train only', 'final_pipeline_fitted_on': 'train + validation',
        'train_rows': len(train), 'validation_rows': len(validation), 'test_used': False,
        'final_fit_last_target_date': combined.target_date.max().strftime('%Y-%m-%d'),
        'numeric_imputation_and_scaling_refitted_with_final_model': True,
        'feature_columns': features,
        'transformed_feature_names': final_pipeline.named_steps['preprocess'].get_feature_names_out().tolist(),
        'random_state': config['random_state'],
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'versions': {'python':platform.python_version(), 'numpy':np.__version__,
                     'pandas':pd.__version__, 'scikit_learn':sklearn.__version__, 'joblib':joblib.__version__},
        'source_sha256': {name: hashlib.sha256((Path(data_dir)/name).read_bytes()).hexdigest()
                          for name in ['train.csv','validation.csv','feature_schema.json']},
    }
    write_json(models / 'selection.json', selection)
    visual_config = config.get('visualization', {})
    if visual_config.get('enabled', True):
        # Pipeline cuối đã được lưu. Các biểu đồ dùng mô hình chẩn đoán mới, fit riêng trên Train.
        try:
            from training_visualization import create_training_visualizations
            manifest = create_training_visualizations(
                train, validation, schema, config, best_each.to_dict(orient='records'), output)
            selection['visualization'] = {
                'successful_curves': manifest['successful_curves'],
                'failed_curves': manifest['failed_curves'],
                'directory': 'results/visualizations',
            }
        except Exception as error:
            selection['visualization'] = {'status': 'failed', 'error': f'{type(error).__name__}: {error}'}
            write_json(results / 'visualization_error.json', selection['visualization'])
            print(f'Mô hình đã lưu; phần biểu đồ gặp lỗi: {error}', flush=True)
        write_json(models / 'selection.json', selection)
    print(f'Pipeline bàn giao: {models / "final_pipeline.joblib"}', flush=True)
    return selection


def main(argv=None):
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--config', type=Path, default=ROOT/'config/training.json')
    parser.add_argument('--output-dir', type=Path, help='Mặc định tạo thư mục mới trong runs/')
    parser.add_argument('--models', nargs='+', choices=MODEL_NAMES)
    parser.add_argument('--quick', action='store_true', help='Chỉ dùng cấu hình đầu tiên của mỗi thuật toán')
    parser.add_argument('--check-data', action='store_true', help='Chỉ kiểm tra dữ liệu và cấu hình, không fit mô hình')
    parser.add_argument('--no-plots', action='store_true', help='Bỏ bước train thêm các mô hình chẩn đoán để vẽ đồ thị')
    args = parser.parse_args(argv)
    try:
        train, validation, schema = load_training_data(args.data_dir)
        config = read_json(args.config)
        from training_visualization import visualization_settings
        config['visualization'] = visualization_settings(config.get('visualization'))
        if args.no_plots:
            config['visualization']['enabled'] = False
        candidates = make_candidates(config, schema, args.models, args.quick)
        if args.check_data:
            print(json.dumps({'train_rows':len(train), 'validation_rows':len(validation),
                              'features_before_encoding':len(schema['feature_columns']),
                              'planned_candidates':len(candidates), 'training_executed':False,
                              'test_loaded':False, 'plots_enabled':config['visualization']['enabled']}, ensure_ascii=False, indent=2))
            return 0
        output = args.output_dir or ROOT/'runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        train_models(train, validation, schema, config, candidates, output, args.data_dir)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f'Không thực hiện được: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
