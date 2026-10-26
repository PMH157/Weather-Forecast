"""Các hàm dùng chung cho phần huấn luyện, không tự train khi import."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

MODEL_NAMES = ['logistic_regression', 'knn', 'decision_tree', 'random_forest', 'svm']
SCALED_MODELS = {'logistic_regression', 'knn', 'svm'}
FORBIDDEN_FEATURES = {'target', 'weather_next_day', 'label_vi', 'weather_today_code',
                      'sample_id', 'origin_date', 'target_date', 'split', 'pressure_raw'}


def configure_console():
    """Xuất UTF-8 để console/VS Code hiển thị tiếng Việt đúng.

    Áp dụng cho cả log, argparse, cảnh báo và stderr trên Windows/VS Code.
    StringIO và các console Unicode không có reconfigure vẫn dùng bình thường.
    Việc đọc/ghi CSV và JSON UTF-8 không bị thay đổi.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, 'reconfigure', None)
        if callable(reconfigure):
            reconfigure(encoding='utf-8', errors='backslashreplace')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def validate_schema(schema):
    features = schema['feature_columns']
    numeric = schema['numeric_features']
    categories = schema['categorical_features']
    if not features or len(features) != len(set(features)):
        raise ValueError('Danh sách đặc trưng trống hoặc có cột trùng')
    if set(features) != set(numeric) | set(categories) or set(numeric) & set(categories):
        raise ValueError('feature_columns không khớp các cột số và phân loại')
    if set(features) & FORBIDDEN_FEATURES:
        raise ValueError('Danh sách X chứa nhãn, ngày định danh hoặc cột chưa được phép sử dụng')
    codes = list(schema['label_to_code'].values())
    if len(codes) != 3 or len(set(codes)) != 3 or any(type(c) is not int for c in codes):
        raise ValueError('Schema phải có đúng ba lớp với ba mã số nguyên khác nhau')


def load_split(path, schema, split_name):
    frame = pd.read_csv(path)
    required = set(schema['feature_columns']) | {'sample_id', 'origin_date', 'target_date', 'target', 'split'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'{Path(path).name} thiếu cột: {sorted(missing)}')
    if frame.empty or not frame['split'].eq(split_name).all():
        raise ValueError(f'{Path(path).name} phải là tập {split_name}, không dùng Test để chọn mô hình')
    for name in ['origin_date', 'target_date']:
        frame[name] = pd.to_datetime(frame[name], format='%Y-%m-%d', errors='raise')
        if frame[name].isna().any():
            raise ValueError(f'Tập {split_name} có ngày bị thiếu')
    if not frame.target_date.sub(frame.origin_date).eq(pd.Timedelta(days=1)).all():
        raise ValueError('Ngày nhãn phải đúng một ngày sau ngày đầu vào')
    if frame.sample_id.isna().any():
        raise ValueError(f'Tập {split_name} có sample_id bị thiếu')
    if frame.duplicated(['city', 'origin_date']).any() or frame.sample_id.duplicated().any():
        raise ValueError(f'Tập {split_name} có mẫu trùng')
    y = pd.to_numeric(frame.target, errors='raise')
    valid_codes = set(schema['label_to_code'].values())
    if y.isna().any() or not y.isin(valid_codes).all():
        raise ValueError('Nhãn thiếu hoặc không có trong ánh xạ của thành viên 2')
    frame['target'] = y.astype(int)
    for name in schema['numeric_features']:
        frame[name] = pd.to_numeric(frame[name], errors='raise')
        if np.isinf(frame[name].to_numpy(dtype=float)).any():
            raise ValueError(f'Cột {name} có infinity; cần xử lý lại ở phần 2')
    for name in schema['categorical_features']:
        if frame[name].isna().any() or frame[name].astype(str).str.strip().eq('').any():
            raise ValueError(f'Cột phân loại {name} bị thiếu')
    return frame.sort_values(['target_date', 'city']).reset_index(drop=True)


def load_training_data(data_dir):
    data_dir = Path(data_dir)
    schema = read_json(data_dir / 'feature_schema.json')
    validate_schema(schema)
    train = load_split(data_dir / 'train.csv', schema, 'train')
    validation = load_split(data_dir / 'validation.csv', schema, 'validation')
    if set(train.target.unique()) != set(schema['label_to_code'].values()):
        raise ValueError('Train chưa có đủ ba lớp; không dùng Test để sửa phân bố lớp')
    empty_features = [name for name in schema['numeric_features'] if train[name].isna().all()]
    if empty_features:
        raise ValueError('Train không có giá trị quan sát cho các đặc trưng: ' + ', '.join(empty_features))
    if set(train.sample_id) & set(validation.sample_id):
        raise ValueError('Train và Validation trùng sample_id')
    if train.target_date.max() > validation.origin_date.min():
        raise ValueError('Nhãn Train vượt qua ngày xuất phát đầu tiên của Validation')
    if set(train.target_date) & set(validation.target_date):
        raise ValueError('Train và Validation có ngày cần dự báo bị trùng')
    return train, validation, schema


def build_pipeline(model_name, schema, random_state=42, n_jobs=-1):
    classifiers = {
        'logistic_regression': LogisticRegression(max_iter=3000, random_state=random_state),
        'knn': KNeighborsClassifier(n_jobs=n_jobs),
        'decision_tree': DecisionTreeClassifier(random_state=random_state),
        'random_forest': RandomForestClassifier(random_state=random_state, n_jobs=n_jobs),
        'svm': SVC(probability=False, cache_size=512, random_state=random_state),
    }
    if model_name not in classifiers:
        raise ValueError(f'Thuật toán chưa hỗ trợ: {model_name}')
    numeric_steps = [('imputer', SimpleImputer(strategy='median', keep_empty_features=True))]
    if model_name in SCALED_MODELS:
        numeric_steps.append(('scaler', StandardScaler()))
    preprocess = ColumnTransformer([
        ('numeric', Pipeline(numeric_steps), schema['numeric_features']),
        ('city', OneHotEncoder(handle_unknown='ignore', sparse_output=False), schema['categorical_features']),
    ], remainder='drop', sparse_threshold=0.0)
    return Pipeline([('preprocess', preprocess), ('classifier', classifiers[model_name])])


def score_predictions(y_true, y_pred, codes):
    return {
        'macro_f1': float(f1_score(y_true, y_pred, labels=codes, average='macro', zero_division=0)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_precision': float(precision_score(y_true, y_pred, labels=codes, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, labels=codes, average='macro', zero_division=0)),
    }


def baseline_predictions(train, validation, codes):
    counts = train.target.value_counts().sort_index()
    majority = int(counts.idxmax())
    result = {'majority_train': np.full(len(validation), majority, dtype=int)}
    if 'weather_today_code' not in validation:
        raise ValueError('Thiếu weather_today_code để tính baseline thời tiết hôm nay')
    today = pd.to_numeric(validation.weather_today_code, errors='coerce').to_numpy(dtype=float)
    result['persistence_today'] = np.where(np.isin(today, codes), today, majority).astype(int)
    return result


def ranked_candidates(records):
    valid = [row for row in records if row.get('status') == 'ok']
    # Ưu tiên Macro F1, sau đó Accuracy trên Validation, cuối cùng mã cấu hình cố định.
    return sorted(valid, key=lambda row: (-row['macro_f1'], -row['accuracy'], row['candidate_id']))
