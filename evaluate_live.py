"""Đo độ chính xác của mô hình loại thời tiết (học từ data/weather.csv) khi nhận số liệu Open-Meteo,
như trang web thời gian thực. Chấm trên đúng giai đoạn Test của Kaggle, so với nhãn Kaggle.

Chạy (sau train.py): python evaluate_live.py [--run RUN_DIR] [--kaggle data/weather.csv] [--open-meteo data/open_meteo.csv]
Ghi kết quả vào RUN_DIR/results/live_weather_accuracy.csv.
"""
import argparse
from pathlib import Path
import sys

import joblib
import pandas as pd

from features import HORIZONS, build_dataset, configure_console, load_weather, make_features
from predict import list_runs, pick_run, read_metadata

ROOT = Path(__file__).resolve().parent


def live_weather_accuracy(run_dir, kaggle_wide, om_wide):
    """Mỗi h: accuracy khi đầu vào là Kaggle (như đồ án) và khi đầu vào là Open-Meteo, trên cùng các ngày Test."""
    columns = read_metadata(run_dir)['feature_columns']
    rows = []
    for h in HORIZONS:
        model = joblib.load(Path(run_dir) / 'models' / f'weather_h{h}.joblib')
        X_k, T_k, split = build_dataset(kaggle_wide, h)
        test = X_k.index[split == 'test']
        X_om = make_features(om_wide, h).reindex(test).dropna(subset=['max'])
        y = T_k.loc[X_om.index, f'weather_h{h}'].astype(int)
        rows.append({'h': h, 'days': len(X_om),
                     'accuracy_kaggle_input': float((model.predict(X_k.loc[X_om.index, columns]) == y).mean()),
                     'accuracy_open_meteo_input': float((model.predict(X_om[columns]) == y).mean())})
    return pd.DataFrame(rows)


def main(argv=None):
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--kaggle', type=Path, default=ROOT / 'data/weather.csv')
    parser.add_argument('--open-meteo', type=Path, default=ROOT / 'data/open_meteo.csv')
    args = parser.parse_args(argv)
    run_dir = args.run or pick_run(list_runs(), 'weather', 'weather.csv')
    if run_dir is None:
        print('Không thực hiện được: chưa có mô hình loại thời tiết; chạy python train.py trước', file=sys.stderr)
        return 2
    result = live_weather_accuracy(run_dir, load_weather(args.kaggle), load_weather(args.open_meteo))
    path = Path(run_dir) / 'results' / 'live_weather_accuracy.csv'
    result.to_csv(path, index=False, encoding='utf-8-sig')
    shown = result.copy()
    for col in ['accuracy_kaggle_input', 'accuracy_open_meteo_input']:
        shown[col] = (100 * shown[col]).round(1)
    print(shown.to_string(index=False))
    print(f'Đã lưu: {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
