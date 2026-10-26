"""Learning/loss curves và các điểm mốc huấn luyện, không tự train khi import.

Các đường học tập dùng prefix ngày lịch của Train và một Validation cố định.
Không đọc Test, không thay đổi mô hình cuối hoặc thứ tự lựa chọn cấu hình.
Chạy không tham số để mở biểu đồ đã lưu, không fit thêm mô hình.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score, log_loss

from modeling import build_pipeline, configure_console, load_training_data, read_json, write_json

ROOT = Path(__file__).resolve().parent
DEFAULT_VISUALIZATION = {
    'enabled': True,
    'learning_curves': True,
    'learning_fractions': [0.10, 0.25, 0.50, 0.75, 1.0],
    'logistic_iteration_curve': True,
    'logistic_iteration_checkpoints': [1, 2, 5, 10, 20, 50, 100, 200, 500],
    'random_forest_tree_curve': True,
    'random_forest_tree_checkpoints': [10, 25, 50, 100, 150, 200],
    'dpi': 160,
}


def visualization_settings(overrides=None):
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError('visualization phải là một đối tượng JSON')
    config = deepcopy(DEFAULT_VISUALIZATION)
    config.update(overrides or {})
    for key in ['enabled', 'learning_curves', 'logistic_iteration_curve', 'random_forest_tree_curve']:
        if type(config[key]) is not bool:
            raise ValueError(f'visualization.{key} phải là true/false')
    if not isinstance(config['learning_fractions'], list):
        raise ValueError('learning_fractions phải là một danh sách')
    try:
        fractions = [float(value) for value in config['learning_fractions']]
    except (TypeError, ValueError) as error:
        raise ValueError('learning_fractions phải chứa các số trong khoảng (0, 1]') from error
    if not fractions or any(not np.isfinite(v) or v <= 0 or v > 1 for v in fractions):
        raise ValueError('learning_fractions phải nằm trong khoảng (0, 1]')
    config['learning_fractions'] = sorted(set(fractions + [1.0]))
    for key in ['logistic_iteration_checkpoints', 'random_forest_tree_checkpoints']:
        values = config[key]
        if not isinstance(values, list) or not values or any(type(v) is not int or v < 1 for v in values):
            raise ValueError(f'{key} phải là danh sách số nguyên dương')
        config[key] = sorted(set(values))
    if type(config['dpi']) is not int or config['dpi'] <= 0:
        raise ValueError('dpi phải là số nguyên dương')
    return config


def chronological_prefixes(train, fractions, codes, minimum_rows=1):
    """Cùng ngày luôn nằm cùng phía mốc; không chia ngẫu nhiên hoặc chọn bằng y Validation."""
    frame = train.sort_values(['target_date', 'city']).reset_index(drop=True)
    dates = np.sort(frame.target_date.unique())
    result, seen = [], set()
    for fraction in sorted(set(list(fractions) + [1.0])):
        n_dates = max(1, int(np.ceil(float(fraction) * len(dates))))
        cutoff = pd.Timestamp(dates[min(n_dates, len(dates)) - 1])
        prefix = frame[frame.target_date.le(cutoff)].copy()
        if len(prefix) in seen or len(prefix) < minimum_rows or set(prefix.target.unique()) != set(codes):
            continue
        seen.add(len(prefix))
        result.append((float(fraction), cutoff, prefix))
    return result


def curve_metrics(pipeline, X_train, y_train, X_val, y_val, codes):
    train_prediction = pipeline.predict(X_train)
    val_prediction = pipeline.predict(X_val)
    metrics = {
        'train_accuracy': float(accuracy_score(y_train, train_prediction)),
        'validation_accuracy': float(accuracy_score(y_val, val_prediction)),
        'train_macro_f1': float(f1_score(y_train, train_prediction, labels=codes, average='macro', zero_division=0)),
        'validation_macro_f1': float(f1_score(y_val, val_prediction, labels=codes, average='macro', zero_division=0)),
    }
    if hasattr(pipeline, 'predict_proba'):
        # predict_proba sắp cột theo classes_ của estimator; đưa về đúng thứ tự codes.
        classes = list(pipeline.classes_)
        order = [classes.index(code) for code in codes]
        train_proba = pipeline.predict_proba(X_train)[:, order]
        val_proba = pipeline.predict_proba(X_val)[:, order]
        metrics.update({
            'loss_name': 'log_loss',
            'train_loss': float(log_loss(y_train, train_proba, labels=codes)),
            'validation_loss': float(log_loss(y_val, val_proba, labels=codes)),
        })
    else:
        # SVC probability=False: không tạo xác suất giả từ decision_function.
        metrics.update({'loss_name': 'classification_error',
                        'train_loss': 1.0 - metrics['train_accuracy'],
                        'validation_loss': 1.0 - metrics['validation_accuracy']})
    return metrics


def save_curve_plot(frame, path, title, x_column, x_label, fit_time_label, dpi=160):
    # Import trễ để --check-data không khởi tạo Matplotlib hoặc train thêm.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    frame = frame.sort_values(x_column)
    x = frame[x_column].to_numpy(dtype=float)
    loss_label = 'Predictive log loss' if frame.loss_name.iloc[0] == 'log_loss' else 'Classification error (1 - accuracy)'
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = [
        (axes[0, 0], 'train_accuracy', 'validation_accuracy', 'Accuracy'),
        (axes[0, 1], 'train_macro_f1', 'validation_macro_f1', 'Macro F1'),
        (axes[1, 0], 'train_loss', 'validation_loss', loss_label),
    ]
    try:
        for axis, train_col, val_col, ylabel in panels:
            axis.plot(x, frame[train_col], 'o-', label='Train', color='#2563eb')
            axis.plot(x, frame[val_col], 'o-', label='Validation', color='#ea580c')
            axis.set_xlabel(x_label)
            axis.set_ylabel(ylabel)
            if ylabel in ['Accuracy', 'Macro F1']:
                axis.set_ylim(0, 1.03)
            else:
                axis.set_ylim(bottom=0)
            axis.grid(alpha=0.25)
            axis.legend()
        axes[1, 1].plot(x, frame.fit_seconds, 'o-', color='#475569')
        axes[1, 1].set_xlabel(x_label)
        axes[1, 1].set_ylabel(fit_time_label)
        axes[1, 1].grid(alpha=0.25)
        fig.suptitle(title, fontsize=15)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, facecolor='white', bbox_inches='tight', pad_inches=0.15)
    finally:
        plt.close(fig)


def learning_curve_rows(model_name, params, train, validation, schema, config, settings):
    features = schema['feature_columns']
    codes = sorted(schema['label_to_code'].values())
    prototype = build_pipeline(model_name, schema, config['random_state'], config['n_jobs'])
    prototype.set_params(**params)
    minimum_rows = prototype.named_steps['classifier'].get_params().get('n_neighbors', 1)
    rows = []
    for fraction, cutoff, prefix in chronological_prefixes(train, settings['learning_fractions'], codes, minimum_rows):
        pipeline = build_pipeline(model_name, schema, config['random_state'], config['n_jobs'])
        pipeline.set_params(**params)
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            # Mỗi prefix học trung vị/scaler riêng; không dùng các thống kê của toàn Train.
            pipeline.fit(prefix[features], prefix.target.to_numpy())
        fit_seconds = time.perf_counter() - start
        rows.append({'requested_fraction': fraction, 'n_train_rows': len(prefix),
                     'last_train_target_date': cutoff.strftime('%Y-%m-%d'), 'fit_seconds': fit_seconds,
                     'warnings': ' | '.join(str(w.message) for w in captured),
                     **curve_metrics(pipeline, prefix[features], prefix.target.to_numpy(),
                                     validation[features], validation.target.to_numpy(), codes)})
    return pd.DataFrame(rows)


def logistic_iteration_rows(params, train, validation, schema, config, settings):
    features = schema['feature_columns']
    codes = sorted(schema['label_to_code'].values())
    prototype = build_pipeline('logistic_regression', schema, config['random_state'], config['n_jobs'])
    prototype.set_params(**params)
    classifier = prototype.named_steps['classifier']
    if classifier.solver != 'lbfgs':
        raise ValueError('Đường iteration này chỉ tái hiện các checkpoint của solver lbfgs')
    budget = int(classifier.max_iter)
    limits = sorted(set([n for n in settings['logistic_iteration_checkpoints'] if n <= budget] + [budget]))
    rows, previous_iterations = [], None
    for limit in limits:
        pipeline = build_pipeline('logistic_regression', schema, config['random_state'], config['n_jobs'])
        pipeline.set_params(**params)
        # Fit lại từ đầu, cùng dữ liệu/solver/tol/seed, tăng giới hạn số vòng lặp.
        # Đây là checkpoint được tái hiện, không phải callback log của một lần fit duy nhất.
        pipeline.set_params(classifier__max_iter=limit, classifier__warm_start=False)
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            pipeline.fit(train[features], train.target.to_numpy())
        seconds = time.perf_counter() - start
        actual = int(np.max(pipeline.named_steps['classifier'].n_iter_))
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in captured)
        if actual != previous_iterations:
            rows.append({'requested_max_iter': limit, 'actual_iterations': actual,
                         'fit_seconds': seconds, 'converged': converged,
                         'checkpoint_method': 'independent fit from the same initialization',
                         'warnings': ' | '.join(str(w.message) for w in captured),
                         **curve_metrics(pipeline, train[features], train.target.to_numpy(),
                                         validation[features], validation.target.to_numpy(), codes)})
            previous_iterations = actual
        if converged:
            break
    return pd.DataFrame(rows)


def random_forest_tree_rows(params, train, validation, schema, config, settings):
    features = schema['feature_columns']
    codes = sorted(schema['label_to_code'].values())
    pipeline = build_pipeline('random_forest', schema, config['random_state'], config['n_jobs'])
    pipeline.set_params(**params)
    classifier = pipeline.named_steps['classifier']
    budget = int(classifier.n_estimators)
    counts = sorted(set([n for n in settings['random_forest_tree_checkpoints'] if n <= budget] + [budget]))
    start = time.perf_counter()
    # Cùng một Train và một bộ biến đổi cố định cho mọi mốc số cây.
    transformed_train = pipeline.named_steps['preprocess'].fit_transform(train[features])
    total_fit_seconds = time.perf_counter() - start
    rows = []
    for count in counts:
        classifier.set_params(n_estimators=count, warm_start=True)
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            classifier.fit(transformed_train, train.target.to_numpy())
        step_seconds = time.perf_counter() - start
        total_fit_seconds += step_seconds
        rows.append({'n_trees': len(classifier.estimators_), 'fit_seconds': total_fit_seconds,
                     'step_fit_seconds': step_seconds,
                     'checkpoint_method': 'warm_start, add trees on unchanged Train',
                     'warnings': ' | '.join(str(w.message) for w in captured),
                     **curve_metrics(pipeline, train[features], train.target.to_numpy(),
                                     validation[features], validation.target.to_numpy(), codes)})
    return pd.DataFrame(rows)


def load_curve_index(run_dir):
    path = Path(run_dir) / 'results/visualizations/curves_index.csv'
    frame = pd.read_csv(path).fillna('')
    if not {'model', 'curve', 'status'}.issubset(frame.columns):
        raise ValueError(f'Chỉ mục biểu đồ thiếu cột bắt buộc: {path}')
    return frame.to_dict(orient='records')


def latest_run_with_failed_curves(runs_dir):
    indices = sorted(Path(runs_dir).glob('*/results/visualizations/curves_index.csv'),
                     key=lambda path: path.stat().st_mtime_ns, reverse=True)
    for index in indices:
        run = index.parents[2]
        required = ['models/selection.json', 'results/best_by_model.csv', 'results/training_config.json']
        if not all((run / name).is_file() for name in required):
            continue
        if any(row['status'] == 'failed' for row in load_curve_index(run)):
            return run
    raise FileNotFoundError('Chưa tìm thấy lần chạy có đủ cấu hình và biểu đồ lỗi trong thư mục runs.')


def create_training_visualizations(train, validation, schema, config, best_by_model, output_dir,
                                   overrides=None, retry_failed=False):
    configure_console()
    settings = visualization_settings(overrides if overrides is not None else config.get('visualization'))
    directory = Path(output_dir) / 'results/visualizations'
    directory.mkdir(parents=True, exist_ok=True)
    index = load_curve_index(output_dir) if retry_failed else []
    failed_keys = {(row['model'], row['curve']) for row in index if row['status'] == 'failed'}
    if retry_failed and not failed_keys:
        raise ValueError('Lần chạy này không có biểu đồ lỗi cần chạy lại.')
    if retry_failed and not settings['enabled']:
        raise ValueError('Trực quan hóa đang tắt; dùng --settings config/training.json để bật khi chạy lại.')
    positions = {(row['model'], row['curve']): number for number, row in enumerate(index)}
    attempted = 0

    def record_result(row):
        key = (row['model'], row['curve'])
        if key in positions:
            index[positions[key]] = row
        else:
            positions[key] = len(index)
            index.append(row)

    for row in best_by_model if settings['enabled'] else []:
        name = row['model']
        params = json.loads(row['params']) if isinstance(row['params'], str) else dict(row['params'])
        tasks = []
        if settings['learning_curves']:
            tasks.append(('learning_curve', learning_curve_rows, 'n_train_rows',
                          'Training rows (chronological prefix)', 'Fit duration (s)'))
        if name == 'logistic_regression' and settings['logistic_iteration_curve']:
            tasks.append(('iteration_checkpoints', logistic_iteration_rows, 'actual_iterations',
                          'Actual L-BFGS iterations at checkpoint', 'Fit duration per checkpoint (s)'))
        if name == 'random_forest' and settings['random_forest_tree_curve']:
            tasks.append(('tree_growth', random_forest_tree_rows, 'n_trees',
                          'Number of trees', 'Cumulative fit time (s)'))
        for kind, function, x_column, x_label, time_label in tasks:
            if retry_failed and (name, kind) not in failed_keys:
                continue
            attempted += 1
            stem = f'{name}_{kind}'
            print(f'Trực quan hóa: {stem}', flush=True)
            try:
                if kind == 'learning_curve':
                    frame = function(name, params, train, validation, schema, config, settings)
                else:
                    frame = function(params, train, validation, schema, config, settings)
                if frame.empty:
                    raise ValueError('Không đủ các lớp hoặc số mẫu để vẽ đường cong')
                frame.to_csv(directory / f'{stem}.csv', index=False, encoding='utf-8-sig')
                title = name.replace('_', ' ').title() + ' | ' + kind.replace('_', ' ')
                save_curve_plot(frame, directory / f'{stem}.png', title, x_column, x_label, time_label, settings['dpi'])
                record_result({'model': name, 'curve': kind, 'status': 'ok', 'points': len(frame),
                              'csv': f'{stem}.csv', 'figure': f'{stem}.png', 'x_axis': x_column,
                              'loss_name': frame.loss_name.iloc[0], 'error': ''})
            except Exception as error:
                record_result({'model': name, 'curve': kind, 'status': 'failed', 'points': 0,
                              'csv': '', 'figure': '', 'x_axis': x_column, 'loss_name': '',
                              'error': f'{type(error).__name__}: {error}'})
                (directory / f'{stem}_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
                print(f'  Biểu đồ gặp lỗi: {error}', flush=True)
    if retry_failed and attempted == 0:
        raise ValueError('Không có đường cong lỗi nào đang bật và khớp cấu hình best_by_model.csv.')
    columns = ['model', 'curve', 'status', 'points', 'csv', 'figure', 'x_axis', 'loss_name', 'error']
    pd.DataFrame(index, columns=columns).to_csv(directory / 'curves_index.csv', index=False, encoding='utf-8-sig')
    manifest = {
        'enabled': settings['enabled'],
        'successful_curves': sum(row['status'] == 'ok' for row in index),
        'failed_curves': sum(row['status'] == 'failed' for row in index),
        'retry_failed_only': retry_failed, 'attempted_curves': attempted,
        'settings': settings, 'test_loaded': False,
        'final_pipeline_modified': False,
        'selection_changed_by_diagnostics': False,
        'notes': [
            'Learning curves use expanding chronological Train prefixes and fixed Validation.',
            'Logistic iteration checkpoints are independent refits, not a live epoch callback.',
            'Random Forest grows trees via warm_start on unchanged Train.',
            'Predictive log loss excludes regularization and is not the complete optimizer objective.',
            'SVM probability=False uses classification error, not an invented probability loss.',
            'Diagnostics fit new models with the selected parameters and do not update the saved final Pipeline.',
        ],
        'index_file': 'results/visualizations/curves_index.csv',
    }
    write_json(directory / 'visualization_manifest.json', manifest)
    return manifest


def main(argv=None):
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, help='Lần chạy để tạo lại các đường cong; bỏ trống để mở biểu đồ đã có')
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--settings', type=Path, help='JSON visualization hoặc JSON training có khóa visualization')
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument('--view-only', action='store_true', help='Chỉ xem biểu đồ đã có, không fit thêm mô hình')
    operation.add_argument('--retry-failed', action='store_true', help='Chỉ tạo lại biểu đồ lỗi; tự tìm lần có lỗi nếu bỏ --run-dir')
    parser.add_argument('--show-plots', action='store_true', help='Mở trang xem sau khi tạo lại các đường cong')
    args = parser.parse_args(argv)
    try:
        if args.retry_failed and args.run_dir is None:
            args.run_dir = latest_run_with_failed_curves(ROOT / 'runs')
            print(f'Chạy lại biểu đồ lỗi của: {args.run_dir}', flush=True)
        if args.run_dir is None or args.view_only:
            from view_training import open_training_gallery
            open_training_gallery(args.run_dir)
            return 0
        selection = read_json(args.run_dir/'models/selection.json')
        for name, expected in selection.get('source_sha256', {}).items():
            if name in ['train.csv', 'validation.csv', 'feature_schema.json']:
                actual = hashlib.sha256((args.data_dir/name).read_bytes()).hexdigest()
                if actual != expected:
                    raise ValueError(f'{name} khác dữ liệu của lần train đã chọn')
        train, validation, schema = load_training_data(args.data_dir)
        config = read_json(args.run_dir/'results/training_config.json')
        best = pd.read_csv(args.run_dir/'results/best_by_model.csv').to_dict(orient='records')
        settings = read_json(args.settings) if args.settings else config.get('visualization', {})
        settings = settings.get('visualization', settings)
        manifest = create_training_visualizations(train, validation, schema, config, best, args.run_dir,
                                                   settings, retry_failed=args.retry_failed)
        print(json.dumps({'successful_curves': manifest['successful_curves'],
                          'failed_curves': manifest['failed_curves']}, ensure_ascii=False))
        if args.show_plots and manifest['successful_curves']:
            from view_training import open_training_gallery
            open_training_gallery(args.run_dir)
        return 0 if manifest['failed_curves'] == 0 else 1
    except (OSError, ValueError, KeyError) as error:
        print(f'Không tạo được biểu đồ: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
