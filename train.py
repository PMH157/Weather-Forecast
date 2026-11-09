"""Huấn luyện 12 mô hình Random Forest (4 đại lượng x 3 ngày dự báo).

Chạy: python train.py [--data data/weather.csv] [--out runs/<thời_điểm>] [--quick]
      python train.py --final-only runs/<thời_điểm>   (chỉ fit lại và chấm Test cho một lần chạy đã tìm tham số)
Thứ tự: tìm tham số trên Train/Validation -> fit lại trên Train + Validation (80%) -> chấm Test (20%) đúng một lần.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from sklearn.compose import ColumnTransformer, make_column_selector  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error  # noqa: E402
from sklearn.model_selection import ParameterGrid  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

import sklearn  # noqa: E402

from features import (CITIES, GAP_DAYS, HORIZONS, TARGETS, build_dataset, split_dates,  # noqa: E402
                      configure_console, load_weather)

ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 42
# Lưới tham số (mục 5.4 của thiết kế): 2 x 3 x 2 = 12 tổ hợp cho mỗi mô hình.
GRID = {'n_estimators': [600], 'max_depth': [None, 10], 'min_samples_leaf': [1, 5, 20],
        'max_features': ['sqrt', 0.5], 'max_samples': [0.6]}
LEAF_BOUNDS = (1, 80)
LEAF_CURVE = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80]
PERSISTENCE_COLUMN = {'weather': 'code_today', 'tmax': 'max', 'tmin': 'min', 'rain': 'rain'}
CONFIDENT_SHARE = 0.7


def kind_of(target):
    return 'clf' if target == 'weather' else 'reg'


def make_pipeline(kind, params):
    """Điền trung vị cho cột số, one-hot cho city, rồi Random Forest."""
    model = (RandomForestClassifier if kind == 'clf' else RandomForestRegressor)(
        random_state=RANDOM_STATE, n_jobs=-1, **params)
    preprocess = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), make_column_selector(dtype_include='number')),
        ('city', OneHotEncoder(handle_unknown='ignore'), ['city']),
    ])
    return Pipeline([('preprocess', preprocess), ('rf', model)])


def _target_month(X, h):
    return (X.index.get_level_values('origin_date') + pd.Timedelta(days=h)).month


def baselines(X_tr, y_tr, X_ev, target, h):
    """'persistence': giá trị hôm nay; 'monthly': nhãn phổ biến nhất / trung bình của (thành phố, tháng ngày dự báo),
    chỉ học từ Train. Tổ hợp chưa thấy trong Train dùng nhãn phổ biến nhất / trung bình chung của Train."""
    y_tr = pd.Series(np.asarray(y_tr, dtype=float), index=X_tr.index)
    keys_tr = pd.MultiIndex.from_arrays([X_tr['city'].to_numpy(), _target_month(X_tr, h)])
    grouped = y_tr.groupby(keys_tr)
    if target == 'weather':
        table, fallback = grouped.agg(lambda s: s.mode().iloc[0]), y_tr.mode().iloc[0]
    else:
        table, fallback = grouped.mean(), y_tr.mean()
    keys_ev = pd.MultiIndex.from_arrays([X_ev['city'].to_numpy(), _target_month(X_ev, h)])
    monthly = table.reindex(keys_ev).fillna(fallback).to_numpy()
    return {'persistence': X_ev[PERSISTENCE_COLUMN[target]].to_numpy(dtype=float), 'monthly': monthly}


def score(target, y_true, y_pred):
    if target == 'weather':
        return {'accuracy': float(accuracy_score(y_true, y_pred)),
                'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0))}
    return {'mae': float(mean_absolute_error(y_true, y_pred))}


def primary(target, metrics):
    return metrics['accuracy'] if target == 'weather' else metrics['mae']


def is_better(target, a, b):
    """Accuracy càng cao càng tốt; MAE càng thấp càng tốt."""
    return a > b if target == 'weather' else a < b


def evaluate(target, params, X_tr, y_tr, X_va, y_va, cache=None):
    """Fit trên Train, chấm Validation. cache tránh fit lại cùng một bộ tham số."""
    key = json.dumps(params, sort_keys=True)
    if cache is not None and key in cache:
        return cache[key]
    pipe = make_pipeline(kind_of(target), params).fit(X_tr, y_tr)
    pred = pipe.predict(X_va)
    result = {'params': params, 'metrics': score(target, y_va, pred), 'pred': pred,
              'proba': pipe.predict_proba(X_va) if target == 'weather' else None}
    if cache is not None:
        cache[key] = result
    return result


def grid_search(target, X_tr, y_tr, X_va, y_va, grid=GRID, cache=None):
    """Brute Force trên lưới tham số; trả về bảng xếp hạng (tốt nhất ở dòng đầu)."""
    rows = []
    for params in ParameterGrid(grid):
        start = time.perf_counter()
        r = evaluate(target, dict(params), X_tr, y_tr, X_va, y_va, cache)
        rows.append({'params': json.dumps(r['params'], sort_keys=True), **r['metrics'],
                     'fit_seconds': round(time.perf_counter() - start, 2)})
    table = pd.DataFrame(rows)
    order = ['accuracy', 'macro_f1'] if target == 'weather' else ['mae']
    return table.sort_values(order, ascending=target != 'weather', kind='stable').reset_index(drop=True)


def golden_leaf(target, base_params, X_tr, y_tr, X_va, y_va, cache=None):
    """Golden Section (scipy, bounded) cho min_samples_leaf trong LEAF_BOUNDS, các tham số khác giữ nguyên.
    Leaf làm tròn thành số nguyên; mỗi leaf chỉ fit một lần."""
    curve = {}

    def objective(x):
        leaf = int(round(x))
        if leaf not in curve:
            r = evaluate(target, {**base_params, 'min_samples_leaf': leaf}, X_tr, y_tr, X_va, y_va, cache)
            curve[leaf] = primary(target, r['metrics'])
        return -curve[leaf] if target == 'weather' else curve[leaf]

    minimize_scalar(objective, bounds=LEAF_BOUNDS, method='bounded', options={'xatol': 1})
    best = (max if target == 'weather' else min)(curve, key=curve.get)
    return {'leaf': int(best), 'score': curve[best], 'n_evals': len(curve), 'curve': dict(sorted(curve.items()))}


def rain_binary_accuracy(y_true, y_pred):
    """Accuracy khi chỉ phân biệt Mưa (2) với Không mưa (0, 1)."""
    return float(np.mean((np.asarray(y_true) == 2) == (np.asarray(y_pred) == 2)))


def confidence_threshold(proba_max, keep=CONFIDENT_SHARE):
    """Ngưỡng xác suất giữ lại tỷ lệ `keep` số ngày mô hình tự tin nhất."""
    return float(np.quantile(proba_max, 1 - keep))


def compare_with_baselines(target, h, X_va, y_va, pred, base):
    """Bảng RF và 2 baseline: gộp 3 thành phố và từng thành phố."""
    rows = []
    city = X_va['city'].to_numpy()
    for name, mask in [('Tất cả', np.ones(len(city), bool))] + [(c, city == c) for c in CITIES]:
        rf = primary(target, score(target, y_va[mask], pred[mask]))
        b = {k: primary(target, score(target, y_va[mask], v[mask])) for k, v in base.items()}
        strongest = max(b, key=b.get) if target == 'weather' else min(b, key=b.get)
        rows.append({'target': target, 'h': h, 'city': name, 'metric': 'accuracy' if target == 'weather' else 'mae',
                     'random_forest': rf, 'persistence': b['persistence'], 'monthly': b['monthly'],
                     'strongest_baseline': strongest, 'rf_beats_strongest': is_better(target, rf, b[strongest])})
    return rows


def plot_leaf_curve(curve, h, path):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    leaves = sorted(curve)
    ax.plot(leaves, [100 * curve[k] for k in leaves], marker='o')
    ax.set_xscale('log')
    ax.set_xticks(leaves, [str(k) for k in leaves])
    ax.set_xlabel('min_samples_leaf (thang log)')
    ax.set_ylabel('Accuracy Validation (%)')
    ax.set_title(f'Loại thời tiết, ngày +{h}: lá nhỏ dễ overfit, lá lớn dễ underfit')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def final_masks(split):
    """Mô hình cuối học trên Train + Validation (80%), chấm trên Test (20%)."""
    split = pd.Series(split)
    return split.isin(['train', 'validation']).to_numpy(), (split == 'test').to_numpy()


def model_path(models_dir, target, h):
    return Path(models_dir) / f'{target}_h{h}.joblib'


def ensure_not_tested(out):
    """Test chỉ được chấm một lần: từ chối nếu lần chạy này đã có mô hình cuối."""
    models = Path(out) / 'models'
    if models.exists() and any(models.iterdir()):
        raise ValueError(f'{models} đã có mô hình cuối, nghĩa là Test đã được chấm. Không chấm Test lần hai; '
                         'muốn làm lại thì chạy train.py với --out mới')


def feature_importance(pipe):
    names = [n.split('__', 1)[1] for n in pipe.named_steps['preprocess'].get_feature_names_out()]
    return pd.Series(pipe.named_steps['rf'].feature_importances_, index=names).sort_values(ascending=False)


def final_fit_and_test(data, out, selected, thresholds):
    """Fit lại các mô hình đã chọn tham số trên Train + Validation, lưu mô hình, chấm Test một lần, vẽ biểu đồ."""
    ensure_not_tested(out)
    targets = [t for t in TARGETS if f'{t}_h{HORIZONS[0]}' in selected]
    results, figures, models = out / 'results', out / 'figures', out / 'models'
    for folder in (results, figures, models):
        folder.mkdir(parents=True, exist_ok=True)
    wide = load_weather(data)
    rows, extra_rows, feature_columns = [], [], None
    for h in HORIZONS:
        X, T, split = build_dataset(wide, h)
        feature_columns = list(X.columns)
        fit, test = final_masks(split)
        X_fit, X_te = X[fit], X[test]
        for target in targets:
            print(f'[ngày +{h}] {target}: fit lại trên Train + Validation, chấm Test...', flush=True)
            y = T[f'{target}_h{h}']
            y = y.astype(int) if target == 'weather' else y
            y_fit, y_te = y[fit].to_numpy(), y[test].to_numpy()
            pipe = make_pipeline(kind_of(target), selected[f'{target}_h{h}']).fit(X_fit, y_fit)
            joblib.dump(pipe, model_path(models, target, h), compress=3)
            pred = pipe.predict(X_te)
            base = baselines(X_fit, y_fit, X_te, target, h)
            rows += compare_with_baselines(target, h, X_te, y_te, pred, base)
            feature_importance(pipe).rename('importance').to_csv(
                results / f'feature_importance_{target}_h{h}.csv', encoding='utf-8-sig', index_label='feature')
            if target == 'weather':
                labels = ['Nắng hoặc ít mây', 'Nhiều mây', 'Mưa']
                matrix = pd.crosstab(pd.Series(y_te, name='thực tế'), pd.Series(pred, name='dự báo'))
                matrix = matrix.reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
                matrix.index, matrix.columns = labels, labels
                matrix.to_csv(results / f'confusion_h{h}.csv', encoding='utf-8-sig')
                conf = pipe.predict_proba(X_te).max(axis=1)
                sure = conf >= thresholds[h]  # ngưỡng học từ Validation, không chỉnh theo Test
                extra_rows.append({'h': h, 'accuracy_all_days': float((pred == y_te).mean()),
                                   'rain_vs_no_rain_accuracy': rain_binary_accuracy(y_te, pred),
                                   'rain_vs_no_rain_persistence': rain_binary_accuracy(y_te, base['persistence']),
                                   'confidence_threshold': thresholds[h], 'confident_share': float(sure.mean()),
                                   'accuracy_confident_days': float((pred[sure] == y_te[sure]).mean())})
                plot_importance(feature_importance(pipe), h, figures / f'feature_importance_weather_h{h}.png')
    metrics = pd.DataFrame(rows)
    metrics.to_csv(results / 'test_metrics.csv', index=False, encoding='utf-8-sig')
    extra = pd.DataFrame(extra_rows)
    extra.to_csv(results / 'test_extra.csv', index=False, encoding='utf-8-sig')
    plot_by_horizon(metrics, figures)
    metadata = {
        'selected_params': selected, 'confidence_threshold': {str(h): t for h, t in thresholds.items()},
        'feature_columns': feature_columns, 'targets': targets, 'horizons': list(HORIZONS),
        'data_file': Path(data).name,
        'split': {'train_end': str(split_dates(wide)[0].date()), 'validation_end': str(split_dates(wide)[1].date()),
                  'gap_days': GAP_DAYS,
                  'final_fit_on': 'train + validation (80%)', 'test': '20%, scored once'},
        'data_sha256': hashlib.sha256(Path(data).read_bytes()).hexdigest(),
        'created_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                     'scikit_learn': sklearn.__version__, 'joblib': joblib.__version__},
    }
    (models / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print_summary(metrics, extra, pd.DataFrame(), title='Test (20%, chấm một lần)')
    return metrics


def plot_importance(importance, h, path, top=15):
    fig, ax = plt.subplots(figsize=(6, 5))
    importance.head(top)[::-1].plot.barh(ax=ax, color='#3b7dd8')
    ax.set_xlabel('Độ quan trọng (giảm Gini trung bình)')
    ax.set_title(f'Loại thời tiết, ngày +{h}: {top} đặc trưng quan trọng nhất')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_by_horizon(metrics, figures):
    allc = metrics[metrics.city == 'Tất cả']
    names = {'random_forest': 'Random Forest', 'persistence': 'Giống hôm nay', 'monthly': 'Theo tháng'}
    w = allc[allc.target == 'weather'].set_index('h')
    if not w.empty:
        plot_weather_by_horizon(w, names, figures)
    numbers = [(t, title) for t, title in [('tmax', 'Nhiệt độ cao (°C)'), ('tmin', 'Nhiệt độ thấp (°C)'),
                                           ('rain', 'Lượng mưa (mm)')] if t in set(allc.target)]
    if not numbers:
        return
    fig, axes = plt.subplots(1, len(numbers), figsize=(4.7 * len(numbers), 4), squeeze=False)
    axes = axes[0]
    for ax, (target, title) in zip(axes, numbers):
        t = allc[allc.target == target].set_index('h')
        for col, label in names.items():
            ax.plot(t.index, t[col], marker='o', label=label)
        ax.set_xticks(list(HORIZONS), [f'+{h}' for h in HORIZONS])
        ax.set_title(f'MAE {title}')
        ax.set_xlabel('Dự báo trước (ngày)')
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(figures / 'mae_by_horizon.png', dpi=150)
    plt.close(fig)


def plot_weather_by_horizon(w, names, figures):
    fig, ax = plt.subplots(figsize=(6, 4))
    for col, label in names.items():
        ax.plot(w.index, 100 * w[col], marker='o', label=label)
    ax.set_xticks(list(HORIZONS), [f'+{h}' for h in HORIZONS])
    ax.set_xlabel('Dự báo trước (ngày)')
    ax.set_ylabel('Accuracy trên Test (%)')
    ax.set_title('Loại thời tiết: accuracy theo số ngày dự báo')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / 'accuracy_by_horizon.png', dpi=150)
    plt.close(fig)


def run(data, out, quick=False, targets=TARGETS):
    results, figures = out / 'results', out / 'figures'
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Thư mục kết quả đã có dữ liệu: {out}; chọn --out khác')
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)
    grid = {k: v[:1] for k, v in GRID.items()} | {'n_estimators': [50]} if quick else GRID
    wide = load_weather(data)
    search_rows, golden_rows, curve_rows, compare_rows, extra_rows = [], [], [], [], []
    selected, thresholds = {}, {}
    for h in HORIZONS:
        X, T, split = build_dataset(wide, h)
        X_tr, X_va = X[split == 'train'], X[split == 'validation']
        for target in targets:
            y = T[f'{target}_h{h}']
            y = y.astype(int) if target == 'weather' else y
            y_tr, y_va = y[split == 'train'].to_numpy(), y[split == 'validation'].to_numpy()
            print(f'[ngày +{h}] {target}: Grid Search {len(ParameterGrid(grid))} tổ hợp...', flush=True)
            cache = {}
            table = grid_search(target, X_tr, y_tr, X_va, y_va, grid, cache)
            search_rows += [{'target': target, 'h': h, 'source': 'grid', **row} for row in table.to_dict('records')]
            best_grid = json.loads(table.loc[0, 'params'])
            grid_score = primary(target, table.loc[0].to_dict())
            chosen = best_grid
            if not quick:
                n_before = len(cache)
                g = golden_leaf(target, {k: v for k, v in best_grid.items() if k != 'min_samples_leaf'},
                                X_tr, y_tr, X_va, y_va, cache)
                golden_rows.append({'target': target, 'h': h, 'grid_best_params': json.dumps(best_grid, sort_keys=True),
                                    'grid_score': grid_score, 'grid_fits': len(table),
                                    'golden_leaf': g['leaf'], 'golden_score': g['score'], 'golden_evals': g['n_evals'],
                                    'golden_new_fits': len(cache) - n_before,
                                    'chosen': 'golden' if is_better(target, g['score'], grid_score) else 'grid'})
                search_rows += [{'target': target, 'h': h, 'source': 'golden',
                                 'params': json.dumps({**best_grid, 'min_samples_leaf': leaf}, sort_keys=True),
                                 ('accuracy' if target == 'weather' else 'mae'): s} for leaf, s in g['curve'].items()]
                if golden_rows[-1]['chosen'] == 'golden':
                    chosen = {**best_grid, 'min_samples_leaf': g['leaf']}
                if target == 'weather':
                    for leaf in LEAF_CURVE:
                        r = evaluate(target, {**chosen, 'min_samples_leaf': leaf}, X_tr, y_tr, X_va, y_va, cache)
                        curve_rows.append({'h': h, 'min_samples_leaf': leaf, **r['metrics']})
                    plot_leaf_curve({r['min_samples_leaf']: r['accuracy'] for r in curve_rows if r['h'] == h},
                                    h, figures / f'leaf_curve_h{h}.png')
            final = evaluate(target, chosen, X_tr, y_tr, X_va, y_va, cache)
            selected[f'{target}_h{h}'] = chosen
            base = baselines(X_tr, y_tr, X_va, target, h)
            compare_rows += compare_with_baselines(target, h, X_va, y_va, final['pred'], base)
            if target == 'weather':
                conf = final['proba'].max(axis=1)
                thresholds[h] = confidence_threshold(conf)
                sure = conf >= thresholds[h]
                extra_rows.append({'h': h, 'accuracy_all_days': final['metrics']['accuracy'],
                                   'rain_vs_no_rain_accuracy': rain_binary_accuracy(y_va, final['pred']),
                                   'rain_vs_no_rain_persistence': rain_binary_accuracy(y_va, base['persistence']),
                                   'confidence_threshold': thresholds[h], 'confident_share': float(sure.mean()),
                                   'accuracy_confident_days': float((final['pred'][sure] == y_va[sure]).mean())})
            pd.DataFrame(search_rows).to_csv(results / 'validation_search.csv', index=False, encoding='utf-8-sig')

    compare = pd.DataFrame(compare_rows)
    compare.to_csv(results / 'validation_vs_baseline.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(extra_rows).to_csv(results / 'validation_extra.csv', index=False, encoding='utf-8-sig')
    if golden_rows:
        pd.DataFrame(golden_rows).to_csv(results / 'golden_vs_grid.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(curve_rows).to_csv(results / 'leaf_curve.csv', index=False, encoding='utf-8-sig')
    (results / 'selected_params.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
    (results / 'confidence_threshold.json').write_text(json.dumps(thresholds, indent=2), encoding='utf-8')
    print_summary(compare, pd.DataFrame(extra_rows), pd.DataFrame(golden_rows))
    if quick:
        # Bản thử nhanh không chấm Test: chạy --quick nhiều lần mà xem Test là đang chỉnh mô hình theo Test.
        print('\n--quick: dừng sau Validation, không fit mô hình cuối và không chấm Test.', flush=True)
    else:
        print('\nĐã chọn xong tham số bằng Validation. Bắt đầu fit lại và chấm Test (một lần).', flush=True)
        final_fit_and_test(data, out, selected, thresholds)
    print(f'\nĐã lưu kết quả tại: {out}')
    return compare


def run_final_only(data, run_dir):
    """Dùng tham số và ngưỡng đã chọn của một lần chạy trước, chỉ fit lại và chấm Test."""
    results = run_dir / 'results'
    try:
        selected = json.loads((results / 'selected_params.json').read_text(encoding='utf-8'))
        thresholds = {int(h): t for h, t in
                      json.loads((results / 'confidence_threshold.json').read_text(encoding='utf-8')).items()}
    except FileNotFoundError as error:
        raise ValueError(f'{run_dir} chưa có kết quả tìm tham số ({error.filename}); chạy python train.py trước')
    final_fit_and_test(data, run_dir, selected, thresholds)
    print(f'\nĐã lưu kết quả tại: {run_dir}')


def print_summary(compare, extra, golden, title='Validation'):
    pd.set_option('display.width', 160)
    allc = compare[compare.city == 'Tất cả'].copy()
    for col in ['random_forest', 'persistence', 'monthly']:
        allc[col] = np.where(allc.metric == 'accuracy', (100 * allc[col]).round(1), allc[col].round(2))
    print(f'\n=== {title}, gộp 3 thành phố (accuracy % cho weather, MAE cho số) ===')
    print(allc[['target', 'h', 'metric', 'random_forest', 'persistence', 'monthly', 'rf_beats_strongest']]
          .to_string(index=False))
    if not extra.empty:
        print('\n=== Chỉ số phụ cho loại thời tiết ===')
        e = extra.copy()
        for col in ['accuracy_all_days', 'rain_vs_no_rain_accuracy', 'rain_vs_no_rain_persistence',
                    'confident_share', 'accuracy_confident_days']:
            e[col] = (100 * e[col]).round(1)
        print(e.round(3).to_string(index=False))
    if not golden.empty:
        print('\n=== Grid Search và Golden Section ===')
        print(golden[['target', 'h', 'grid_score', 'grid_fits', 'golden_leaf', 'golden_score', 'golden_evals',
                      'chosen']].round(4).to_string(index=False))


def main(argv=None):
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/weather.csv')
    parser.add_argument('--out', type=Path, help='Mặc định: runs/<thời_điểm>')
    parser.add_argument('--quick', action='store_true',
                        help='1 tổ hợp, 50 cây, bỏ Golden Section, không chấm Test: chỉ để thử code chạy được')
    parser.add_argument('--final-only', type=Path, metavar='RUN_DIR',
                        help='Bỏ bước tìm tham số; fit lại và chấm Test cho lần chạy RUN_DIR (một lần duy nhất)')
    parser.add_argument('--targets', nargs='+', choices=TARGETS, default=TARGETS,
                        help='Chỉ train các đại lượng này, ví dụ --targets tmax tmin rain')
    args = parser.parse_args(argv)
    out = args.out or ROOT / 'runs' / datetime.now().strftime('%Y%m%d_%H%M%S')
    try:
        if args.final_only:
            run_final_only(args.data, args.final_only)
        else:
            run(args.data, out, args.quick, args.targets)
    except (OSError, ValueError) as error:
        print(f'Không thực hiện được: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
