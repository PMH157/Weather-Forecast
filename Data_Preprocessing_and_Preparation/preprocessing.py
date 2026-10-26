"""Tiền xử lý cho phân loại trạng thái thời tiết ngày kế tiếp.

Thực hiện phần thành viên 2. Không huấn luyện hoặc lựa chọn thuật toán phân loại.
Các biến thống kê tiền xử lý chỉ được học trên Train.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
SOURCE_COLUMNS = ['province', 'max', 'min', 'wind', 'wind_d', 'rain', 'humidi', 'cloud', 'pressure', 'date']
COMPASS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
WIND_DEGREES = {name: i * 22.5 for i, name in enumerate(COMPASS)}
CITY_NAMES = {'Nha Trang': 'Nha Trang', 'Ho Chi Minh City': 'Ho Chi Minh City', 'Ha Noi': 'Ha Noi', 'Hanoi': 'Ha Noi'}
RENAME = {'max': 'temperature_max_c', 'min': 'temperature_min_c', 'wind': 'wind_speed_km_h',
          'rain': 'rain_mm', 'humidi': 'humidity_pct', 'cloud': 'cloud_cover_pct', 'pressure': 'pressure_raw'}
CURRENT = ['temperature_max_c', 'temperature_min_c', 'humidity_pct', 'rain_mm',
           'wind_speed_m_s', 'cloud_cover_pct', 'wind_direction_sin', 'wind_direction_cos']
LAG_BASE = ['temperature_max_c', 'temperature_min_c', 'humidity_pct', 'rain_mm']
ROLL_BASE = ['temperature_max_c', 'temperature_min_c', 'humidity_pct', 'rain_mm', 'cloud_cover_pct', 'wind_speed_m_s']
NUMERIC_FEATURES = CURRENT + [f'{name}_lag{lag}' for lag in [1, 2] for name in LAG_BASE] + [
    f'{name}_rolling3_mean' for name in ROLL_BASE] + ['forecast_day_sin', 'forecast_day_cos']
FEATURE_COLUMNS = ['city'] + NUMERIC_FEATURES
METADATA_COLUMNS = ['sample_id', 'origin_date', 'target_date', 'city', 'weather_today_code',
                    'weather_next_day', 'label_vi', 'target', 'split']
LABELS = {'sunny_or_few_clouds': 0, 'cloudy': 1, 'rain': 2}
LABEL_VI = {'sunny_or_few_clouds': 'Nắng hoặc ít mây', 'cloudy': 'Nhiều mây', 'rain': 'Mưa'}
DEFAULT_CONFIG = {
    'rain_threshold_mm': 1.0, 'cloud_threshold_pct': 50.0,
    'temperature_min_allowed_c': -20.0, 'temperature_max_allowed_c': 50.0,
    'max_wind_speed_m_s': 100.0,
    'train_fraction': 0.60, 'validation_fraction': 0.15,
    'pressure_used_as_feature': False,
    'hanoi_priority_source_name': 'Ha Noi',
    'forecast_origin': 'After the end of day t, using final daily measurements through t',
    'daily_rain_assumption': 'Treat rain as the precipitation amount attached to that source day; operational project labels, not independently verified meteorological observations',
}


def json_write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def csv_write(frame: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding='utf-8-sig', date_format='%Y-%m-%d', lineterminator='\n')


def validate_config(config):
    if not (0 < config['train_fraction'] < 1 and 0 < config['validation_fraction'] < 1
            and config['train_fraction'] + config['validation_fraction'] < 1):
        raise ValueError('Train/Validation fractions must be positive and add to less than 1')
    if not (0 <= config['cloud_threshold_pct'] <= 100) or config['rain_threshold_mm'] <= 0:
        raise ValueError('Invalid label thresholds')
    if config['temperature_min_allowed_c'] >= config['temperature_max_allowed_c']:
        raise ValueError('Invalid temperature quality-control range')
    if config['max_wind_speed_m_s'] <= 0:
        raise ValueError('Invalid wind speed quality-control range')
    if config['pressure_used_as_feature']:
        raise ValueError('Pressure unit is unconfirmed; this version preserves pressure_raw outside model inputs')
    if config['hanoi_priority_source_name'] not in ['Ha Noi', 'Hanoi']:
        raise ValueError('Hanoi source priority must name one of the two original source series')


def make_weather_labels(rain: pd.Series, cloud: pd.Series, config=None) -> pd.Series:
    """Không gán hoặc điền nhãn khi không đủ số liệu; ưu tiên nhãn Mưa."""
    cfg = DEFAULT_CONFIG if config is None else config
    rain = pd.to_numeric(rain, errors='coerce')
    cloud = pd.to_numeric(cloud, errors='coerce')
    labels = pd.Series(pd.NA, index=rain.index, dtype='string')
    valid_rain = np.isfinite(rain) & rain.ge(0)
    rainy = valid_rain & rain.ge(cfg['rain_threshold_mm'])
    labels.loc[rainy] = 'rain'
    non_rainy = valid_rain & rain.lt(cfg['rain_threshold_mm']) & cloud.between(0, 100)
    labels.loc[non_rainy & cloud.ge(cfg['cloud_threshold_pct'])] = 'cloudy'
    labels.loc[non_rainy & cloud.lt(cfg['cloud_threshold_pct'])] = 'sunny_or_few_clouds'
    return labels


def clean_daily(raw: pd.DataFrame, config=None):
    """Chọn nguồn, sửa kiểu, kiểm tra giá trị và dựng lịch ngày thật.

    Giá trị ngoài phạm vi được chuyển thành thiếu, không xóa cả bản ghi.
    Áp suất được giữ nguyên giá trị số trong cột đối chiếu.
    """
    cfg = deepcopy(DEFAULT_CONFIG if config is None else config)
    validate_config(cfg)
    if not set(SOURCE_COLUMNS).issubset(raw.columns):
        raise ValueError('Missing source columns: ' + ', '.join(sorted(set(SOURCE_COLUMNS) - set(raw.columns))))
    df = raw[SOURCE_COLUMNS].copy()
    df['province'] = df.province.astype('string').str.strip()
    audit = {'input_rows': int(len(df)), 'outside_scope_rows': int((~df.province.isin(CITY_NAMES)).sum())}
    df = df[df.province.isin(CITY_NAMES)].copy()
    # Dùng một chuỗi Hà Nội nhất quán, không trộn tùy ngày với chuỗi còn lại.
    if (df.province == cfg['hanoi_priority_source_name']).any():
        other_name = 'Hanoi' if cfg['hanoi_priority_source_name'] == 'Ha Noi' else 'Ha Noi'
        redundant = df.province.eq(other_name)
        audit['hanoi_alternate_series_rows_excluded'] = int(redundant.sum())
        df = df[~redundant].copy()
    else:
        audit['hanoi_alternate_series_rows_excluded'] = 0
    parsed = pd.to_datetime(df.date, format='%Y-%m-%d', errors='coerce')
    invalid_date_rows = df[parsed.isna()].copy()
    audit['invalid_date_rows_excluded'] = len(invalid_date_rows)
    df['date'] = parsed
    df = df[df.date.notna()].copy()
    df['city'] = df.province.map(CITY_NAMES)
    audit['exact_duplicate_rows_removed'] = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    duplicates = df.duplicated(['city', 'date'], keep=False)
    if duplicates.any():
        raise ValueError('Conflicting duplicate city/date rows; resolve source conflict before preprocessing')
    if set(df.city.unique()) != {'Nha Trang', 'Ho Chi Minh City', 'Ha Noi'}:
        raise ValueError('The input must contain all three project cities')
    df = df.rename(columns=RENAME)
    df['source_row_present'] = True
    issues = []
    for column in list(RENAME.values()):
        source_values = df[column].copy()
        numeric = pd.to_numeric(source_values, errors='coerce').astype(float)
        mask = source_values.notna() & (~np.isfinite(numeric))
        for idx in df.index[mask]:
            issues.append({'city': df.at[idx, 'city'], 'date': df.at[idx, 'date'], 'column': column,
                           'original_value': str(source_values.loc[idx]), 'reason': 'non_numeric_or_non_finite'})
        df[column] = numeric.where(np.isfinite(numeric), np.nan)
    df['wind_speed_m_s'] = df.wind_speed_km_h / 3.6
    df['wind_d'] = df.wind_d.astype('string').str.strip().str.upper()
    df['wind_direction_deg'] = df.wind_d.map(WIND_DEGREES).astype(float)
    bad_directions = df.wind_d.notna() & df.wind_direction_deg.isna()
    for idx in df.index[bad_directions]:
        issues.append({'city': df.at[idx, 'city'], 'date': df.at[idx, 'date'], 'column': 'wind_d',
                       'original_value': str(df.at[idx, 'wind_d']), 'reason': 'unknown_compass_direction'})
    checks = {
        'temperature_max_c': df.temperature_max_c.notna() & ~df.temperature_max_c.between(cfg['temperature_min_allowed_c'], cfg['temperature_max_allowed_c']),
        'temperature_min_c': df.temperature_min_c.notna() & ~df.temperature_min_c.between(cfg['temperature_min_allowed_c'], cfg['temperature_max_allowed_c']),
        'humidity_pct': df.humidity_pct.notna() & ~df.humidity_pct.between(0, 100),
        'cloud_cover_pct': df.cloud_cover_pct.notna() & ~df.cloud_cover_pct.between(0, 100),
        'rain_mm': df.rain_mm.notna() & df.rain_mm.lt(0),
        'wind_speed_m_s': df.wind_speed_m_s.notna() & ~df.wind_speed_m_s.between(0, cfg['max_wind_speed_m_s']),
    }
    reversed_temp = df.temperature_max_c.lt(df.temperature_min_c)
    for col in ['temperature_max_c', 'temperature_min_c']:
        checks[col] |= reversed_temp
    for col, mask in checks.items():
        for idx in df.index[mask]:
            issues.append({'city': df.at[idx, 'city'], 'date': df.at[idx, 'date'], 'column': col,
                           'original_value': str(df.at[idx, col]),
                           'reason': 'max_below_min' if reversed_temp.loc[idx] and col.startswith('temperature') else 'outside_project_quality_range'})
        df.loc[mask, col] = np.nan
    df['wind_direction_sin'] = np.sin(np.deg2rad(df.wind_direction_deg))
    df['wind_direction_cos'] = np.cos(np.deg2rad(df.wind_direction_deg))
    calendar_parts = []
    for city, group in df.groupby('city', sort=True):
        grid = group.set_index('date').sort_index().reindex(pd.date_range(group.date.min(), group.date.max(), freq='D'))
        grid.index.name = 'date'
        grid['city'] = city
        grid['source_row_present'] = grid.source_row_present.eq(True)
        calendar_parts.append(grid.reset_index())
    daily = pd.concat(calendar_parts, ignore_index=True).sort_values(['city', 'date']).reset_index(drop=True)
    daily['weather_today'] = make_weather_labels(daily.rain_mm, daily.cloud_cover_pct, cfg)
    daily['weather_today_code'] = daily.weather_today.map(LABELS).astype('Int64')
    audit.update({'selected_observed_rows': int(daily.source_row_present.sum()), 'calendar_rows': len(daily),
                  'inserted_calendar_rows': int((~daily.source_row_present).sum()),
                  'flagged_measurement_cells': len(issues),
                  'pressure_used_as_feature': False, 'numeric_imputation_at_cleaning': False})
    issue_frame = pd.DataFrame(issues, columns=['city', 'date', 'column', 'original_value', 'reason'])
    return daily, audit, issue_frame, invalid_date_rows


def build_supervised(daily: pd.DataFrame, config=None):
    """Đặc trưng chỉ lấy ngày t hoặc trước đó; nhãn lấy đúng ngày t+1."""
    cfg = DEFAULT_CONFIG if config is None else config
    frame = daily.sort_values(['city', 'date']).copy()
    for _, g in frame.groupby('city', sort=False):
        if not g.date.diff().dropna().eq(pd.Timedelta(days=1)).all():
            raise ValueError('Feature building requires a complete daily calendar for each city')
    group = frame.groupby('city', sort=False)
    for lag in [1, 2]:
        for col in LAG_BASE:
            frame[f'{col}_lag{lag}'] = group[col].shift(lag)
    for col in ROLL_BASE:
        frame[f'{col}_rolling3_mean'] = group[col].transform(lambda s: s.rolling(3, min_periods=3).mean())
    frame['origin_date'] = frame.date
    frame['target_date'] = frame.date + pd.Timedelta(days=1)
    # Calendar hoàn chỉnh giúp shift(-1) luôn nghĩa là ngày mai, không nhảy ngày bị thiếu.
    frame['weather_next_day'] = group.weather_today.shift(-1)
    days_in_forecast_year = np.where(frame.target_date.dt.is_leap_year, 366.0, 365.0)
    angle = 2 * np.pi * (frame.target_date.dt.dayofyear.to_numpy() - 1) / days_in_forecast_year
    frame['forecast_day_sin'], frame['forecast_day_cos'] = np.sin(angle), np.cos(angle)
    observed_origins = frame[frame.source_row_present].copy()
    rejected = observed_origins[observed_origins.weather_next_day.isna()].copy()
    rejected['reason'] = np.where(rejected.origin_date.eq(frame.groupby('city').date.transform('max').loc[rejected.index]),
                                  'next_day_outside_dataset', 'next_day_label_unavailable')
    eligible = observed_origins[observed_origins.weather_next_day.notna()].copy()
    eligible['target'] = eligible.weather_next_day.map(LABELS).astype(int)
    eligible['label_vi'] = eligible.weather_next_day.map(LABEL_VI)
    eligible['sample_id'] = eligible.city.str.replace(' ', '_') + '__' + eligible.origin_date.dt.strftime('%Y-%m-%d')
    eligible['split'] = ''
    eligible = eligible.sort_values(['target_date', 'city']).reset_index(drop=True)
    return eligible[METADATA_COLUMNS + NUMERIC_FEATURES], rejected[['city', 'origin_date', 'target_date', 'reason']]


def split_by_time(dataset, config=None):
    cfg = DEFAULT_CONFIG if config is None else config
    dates = np.sort(dataset.target_date.unique())
    n_train = int(len(dates) * cfg['train_fraction'])
    n_val = int(len(dates) * cfg['validation_fraction'])
    if min(n_train, n_val, len(dates) - n_train - n_val) < 1:
        raise ValueError('Insufficient distinct target dates for all three temporal splits')
    mapping = {pd.Timestamp(day): 'train' if i < n_train else 'validation' if i < n_train + n_val else 'test'
               for i, day in enumerate(dates)}
    result = dataset.copy()
    result['split'] = result.target_date.map(mapping)
    splits = {name: result[result.split.eq(name)].reset_index(drop=True) for name in ['train', 'validation', 'test']}
    summaries = []
    for name, part in splits.items():
        summaries.append({'split': name, 'rows': len(part), 'target_days': int(part.target_date.nunique()),
                          'origin_start': part.origin_date.min().strftime('%Y-%m-%d'),
                          'origin_end': part.origin_date.max().strftime('%Y-%m-%d'),
                          'target_start': part.target_date.min().strftime('%Y-%m-%d'),
                          'target_end': part.target_date.max().strftime('%Y-%m-%d'),
                          'row_fraction_pct': round(len(part) / len(result) * 100, 5)})
    assert splits['train'].target_date.max() <= splits['validation'].origin_date.min()
    assert splits['validation'].target_date.max() <= splits['test'].origin_date.min()
    assert result.groupby('target_date').split.nunique().max() == 1
    return result, splits, pd.DataFrame(summaries)


def make_preprocessor(scale=True):
    steps = [('imputer', SimpleImputer(strategy='median', keep_empty_features=True))]
    if scale:
        steps.append(('scaler', StandardScaler()))
    return ColumnTransformer([
        ('numeric', Pipeline(steps), NUMERIC_FEATURES),
        ('city', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['city']),
    ], remainder='drop', sparse_threshold=0.0, verbose_feature_names_out=True)


def class_distribution(dataset):
    rows = []
    for split in ['train', 'validation', 'test']:
        part = dataset[dataset.split.eq(split)]
        for label, code in LABELS.items():
            n = int(part.target.eq(code).sum())
            rows.append({'split': split, 'label': label, 'label_vi': LABEL_VI[label], 'target': code,
                         'count': n, 'pct_in_split': round(n / len(part) * 100, 5)})
    return pd.DataFrame(rows)


def feature_dictionary():
    rows = []
    unit = {'temperature_max_c': '°C', 'temperature_min_c': '°C', 'humidity_pct': '%',
            'rain_mm': 'mm per source date', 'wind_speed_m_s': 'm/s', 'cloud_cover_pct': '%',
            'wind_direction_sin': 'dimensionless', 'wind_direction_cos': 'dimensionless'}
    rows.append({'feature': 'city', 'type': 'categorical', 'unit_before_scaling': 'city name',
                 'latest_measurement_offset_days': 0, 'meaning_vi': 'Thành phố; One-Hot Encoding học trên Train'})
    for col in CURRENT:
        rows.append({'feature': col, 'type': 'numeric', 'unit_before_scaling': unit[col],
                     'latest_measurement_offset_days': 0, 'meaning_vi': 'Giá trị ngày t; hướng gió được biến đổi sin/cos'})
    for lag in [1, 2]:
        for col in LAG_BASE:
            rows.append({'feature': f'{col}_lag{lag}', 'type': 'numeric', 'unit_before_scaling': unit[col],
                         'latest_measurement_offset_days': -lag, 'meaning_vi': f'Giá trị đúng ngày lịch t-{lag}, không nhảy qua ngày thiếu'})
    for col in ROLL_BASE:
        rows.append({'feature': f'{col}_rolling3_mean', 'type': 'numeric', 'unit_before_scaling': unit[col],
                     'latest_measurement_offset_days': 0, 'meaning_vi': 'Trung bình ba ngày lịch t-2 đến t; thiếu một ngày thì để NaN trước imputation'})
    for col in ['forecast_day_sin', 'forecast_day_cos']:
        rows.append({'feature': col, 'type': 'numeric', 'unit_before_scaling': 'dimensionless',
                     'latest_measurement_offset_days': None, 'meaning_vi': 'Ngày trong năm của ngày dự báo; lịch biết trước, không phải phép đo tương lai'})
    return pd.DataFrame(rows)


def run(input_path=None, output_dir=None, config=None):
    input_path = Path(input_path) if input_path else ROOT / 'data/interim/weather_raw.csv'
    output = Path(output_dir) if output_dir else ROOT
    cfg = deepcopy(DEFAULT_CONFIG if config is None else config)
    validate_config(cfg)
    dirs = {
        'data/clean': output / 'data/processed/clean',
        'data/splits': output / 'data/processed/splits',
        'data/ready': output / 'data/processed/ready',
        'metadata': output / 'reports/member2',
        'artifacts': output / 'artifacts',
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(input_path)
    daily, audit, issues, invalid_dates = clean_daily(raw, cfg)
    dataset, rejected = build_supervised(daily, cfg)
    dataset, splits, boundaries = split_by_time(dataset, cfg)
    if not set(splits['train'].target.unique()) == set(LABELS.values()):
        raise ValueError('Train must contain all three classes; do not fix this by consulting Test')
    scaled, trees = make_preprocessor(True), make_preprocessor(False)
    X_train = splits['train'][FEATURE_COLUMNS]
    # Không fit, fit_transform hoặc ước lượng thống kê từ Validation/Test.
    scaled.fit(X_train)
    trees.fit(X_train)
    prepared_parts, arrays_scaled, arrays_trees = [], {}, {}
    names_scaled = scaled.get_feature_names_out().astype(str)
    names_trees = trees.get_feature_names_out().astype(str)
    fill_counts = []
    for name, part in splits.items():
        X = part[FEATURE_COLUMNS]
        xs, xt = scaled.transform(X), trees.transform(X)
        if not np.isfinite(xs).all() or not np.isfinite(xt).all():
            raise ValueError('Transformed matrices must contain only finite numeric values')
        current = part[METADATA_COLUMNS].reset_index(drop=True)
        current = pd.concat([current, pd.DataFrame(xs, columns=names_scaled)], axis=1)
        prepared_parts.append(current)
        y = part.target.to_numpy(dtype=np.int64)
        for archive, matrix in [(arrays_scaled, xs), (arrays_trees, xt)]:
            archive[f'X_{name}'] = matrix
            archive[f'y_{name}'] = y
            archive[f'sample_id_{name}'] = part.sample_id.to_numpy(dtype=str)
        csv_write(part, dirs['data/splits'] / f'{name}.csv')
        for col in NUMERIC_FEATURES:
            fill_counts.append({'split': name, 'feature': col, 'missing_before': int(X[col].isna().sum()),
                                'missing_after': 0})
    arrays_scaled['feature_names'], arrays_trees['feature_names'] = names_scaled, names_trees
    np.savez_compressed(dirs['data/ready'] / 'scaled_arrays.npz', **arrays_scaled)
    np.savez_compressed(dirs['data/ready'] / 'tree_arrays.npz', **arrays_trees)
    prepared = pd.concat(prepared_parts, ignore_index=True).sort_values(['target_date', 'city']).reset_index(drop=True)
    csv_write(prepared, dirs['data/ready'] / 'weather_prepared.csv')
    csv_write(dataset, dirs['data/clean'] / 'weather_labeled_features.csv')
    csv_write(daily, dirs['data/clean'] / 'weather_daily_calendar.csv')
    csv_write(rejected, dirs['metadata'] / 'excluded_forecast_origins.csv')
    csv_write(daily.loc[~daily.source_row_present, ['city', 'date']], dirs['metadata'] / 'missing_calendar_days.csv')
    csv_write(issues, dirs['metadata'] / 'invalid_measurement_cells.csv')
    csv_write(invalid_dates, dirs['metadata'] / 'invalid_date_rows.csv')
    csv_write(boundaries, dirs['metadata'] / 'split_boundaries.csv')
    distribution = class_distribution(dataset)
    csv_write(distribution, dirs['metadata'] / 'class_distribution.csv')
    csv_write(dataset.groupby(['city', 'split', 'label_vi']).size().rename('count').reset_index(), dirs['metadata'] / 'class_distribution_by_city.csv')
    csv_write(pd.DataFrame(fill_counts), dirs['metadata'] / 'missing_feature_counts.csv')
    csv_write(feature_dictionary(), dirs['metadata'] / 'feature_dictionary.csv')
    train_counts = splits['train'].groupby('target').size()
    csv_write(train_counts.rename('count').reset_index(), dirs['metadata'] / 'training_class_counts.csv')
    json_write(dirs['metadata'] / 'class_weights_from_train.json', {
        'formula': 'n_train / (n_classes * count_of_class_in_train)',
        'weights': {str(code): float(len(splits['train']) / (len(LABELS) * train_counts[code])) for code in LABELS.values()},
        'computed_from': 'train only', 'applied_to_data': False,
        'note_vi': 'Chỉ là gợi ý cho mô hình có hỗ trợ trọng số lớp; thành viên 3 quyết định bằng Validation.',
    })
    imputer = scaled.named_transformers_['numeric'].named_steps['imputer']
    scaler = scaled.named_transformers_['numeric'].named_steps['scaler']
    stats = [{'feature': col, 'median_train': float(imputer.statistics_[i]),
              'mean_after_train_imputation': float(scaler.mean_[i]), 'std_scale_train': float(scaler.scale_[i])}
             for i, col in enumerate(NUMERIC_FEATURES)]
    csv_write(pd.DataFrame(stats), dirs['metadata'] / 'train_preprocessing_statistics.csv')
    joblib.dump(scaled, dirs['artifacts'] / 'scaled_preprocessor_train.joblib')
    joblib.dump(trees, dirs['artifacts'] / 'tree_preprocessor_train.joblib')
    schema = {
        'feature_columns': FEATURE_COLUMNS, 'numeric_features': NUMERIC_FEATURES,
        'categorical_features': ['city'], 'metadata_columns_never_use_as_features': [c for c in METADATA_COLUMNS if c != 'city'],
        'scaled_feature_names': names_scaled.tolist(), 'tree_feature_names': names_trees.tolist(),
        'label_to_code': LABELS, 'label_display_vi': LABEL_VI,
        'label_rules': {'rain': f'rain >= {cfg["rain_threshold_mm"]} mm in the source daily record',
                        'cloudy': f'rain < {cfg["rain_threshold_mm"]} and cloud >= {cfg["cloud_threshold_pct"]}%',
                        'sunny_or_few_clouds': f'rain < {cfg["rain_threshold_mm"]} and cloud < {cfg["cloud_threshold_pct"]}%'},
        'horizon_days': 1, 'origin_semantics': cfg['forecast_origin'],
        'pressure_in_model': False, 'pressure_note': 'Preserved in the clean calendar only; source unit unresolved',
        'final_classifier_trained': False, 'preprocessing_fit_split': 'train',
    }
    json_write(dirs['metadata'] / 'feature_schema.json', schema)
    json_write(dirs['metadata'] / 'preprocessing_config.json', cfg)
    versions = {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                'scikit_learn': sklearn.__version__, 'joblib': joblib.__version__}
    json_write(dirs['metadata'] / 'runtime_versions.json', versions)
    checks = {
        'target_exactly_one_calendar_day_ahead': bool(dataset.target_date.sub(dataset.origin_date).eq(pd.Timedelta(days=1)).all()),
        'unique_city_origin': not bool(dataset.duplicated(['city', 'origin_date']).any()),
        'same_target_date_same_split_all_cities': bool(dataset.groupby('target_date').split.nunique().eq(1).all()),
        'train_labels_available_by_validation_origin': bool(splits['train'].target_date.max() <= splits['validation'].origin_date.min()),
        'validation_labels_available_by_test_origin': bool(splits['validation'].target_date.max() <= splits['test'].origin_date.min()),
        'imputer_matches_train_medians': bool(np.allclose(imputer.statistics_, X_train[NUMERIC_FEATURES].median().to_numpy())),
        'scaler_fitted_rows_equal_train_rows': bool(int(scaler.n_samples_seen_) == len(X_train)),
        'all_scaled_values_finite': bool(np.isfinite(prepared[names_scaled].to_numpy(dtype=float)).all()),
        'no_label_or_pressure_in_feature_list': not bool(set(FEATURE_COLUMNS) & {'target', 'weather_next_day', 'label_vi', 'weather_today_code', 'pressure_raw'}),
    }
    if not all(checks.values()):
        raise AssertionError('Data preparation integrity check failed')
    json_write(dirs['metadata'] / 'integrity_checks.json', checks)
    summary = {
        'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
        'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'audit': audit,
        'supervised_rows': len(dataset), 'excluded_observed_origins': len(rejected),
        'exclusion_counts': {str(k): int(v) for k, v in rejected.reason.value_counts().items()},
        'numeric_features_before_encoding': len(NUMERIC_FEATURES), 'categorical_features_before_encoding': 1,
        'features_after_one_hot': len(names_scaled), 'split_boundaries': boundaries.to_dict(orient='records'),
        'class_distribution': distribution.to_dict(orient='records'),
        'total_numeric_cells_imputed': int(pd.DataFrame(fill_counts).missing_before.sum()),
        'imputed_cells_by_split': {str(k): int(v) for k, v in pd.DataFrame(fill_counts).groupby('split').missing_before.sum().items()},
        'checks': checks, 'classifier_trained': False,
    }
    json_write(dirs['metadata'] / 'preprocessing_summary.json', summary)
    # In JSON ASCII để tương thích terminal Windows cp1252; file vẫn lưu UTF-8.
    print(json.dumps({k: summary[k] for k in ['supervised_rows', 'excluded_observed_origins', 'features_after_one_hot',
                                            'split_boundaries', 'class_distribution', 'total_numeric_cells_imputed']}, ensure_ascii=True, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/interim/weather_raw.csv')
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/preprocessing.json', help='JSON configuration with overrides to defaults')
    args = parser.parse_args(argv)
    config = deepcopy(DEFAULT_CONFIG)
    if args.config:
        config.update(json.loads(args.config.read_text(encoding='utf-8')))
    run(args.input, args.output, config)


if __name__ == '__main__':
    main()
