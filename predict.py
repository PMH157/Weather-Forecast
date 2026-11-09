"""In bản tin dự báo 3 ngày cho Hà Nội, TP.HCM, Nha Trang.

Thời gian thực (cần Internet):  python predict.py --live
    Lấy số liệu Open-Meteo đến hết hôm qua -> dự báo hôm nay, ngày mai, ngày kia.
    Loại thời tiết: mô hình học từ data/weather.csv; nhiệt độ, lượng mưa: mô hình học từ data/open_meteo.csv.
Từ file CSV:  python predict.py [--data data/weather.csv]   (dự báo 3 ngày sau ngày cuối cùng trong file)
Tuỳ chọn: --run RUN_DIR (mô hình loại thời tiết), --numbers-run RUN_DIR (mô hình con số), --out forecast.csv
"""
import argparse
import json
from pathlib import Path
import sys

import joblib
import pandas as pd

from features import CITIES, HORIZONS, TARGETS, configure_console, load_weather, make_features  # noqa: F401

ROOT = Path(__file__).resolve().parent
WEATHER_VI = {0: 'Nắng hoặc ít mây', 1: 'Nhiều mây', 2: 'Mưa'}


WEEKDAY = ['Thứ Hai', 'Thứ Ba', 'Thứ Tư', 'Thứ Năm', 'Thứ Sáu', 'Thứ Bảy', 'Chủ Nhật']


def day_label(target_date, today=None):
    """'Hôm nay, Chủ Nhật 09/11' khi biết hôm nay là ngày nào; ngược lại chỉ 'Thứ Bảy 19/06'."""
    text = f'{WEEKDAY[target_date.weekday()]} {target_date:%d/%m}'
    relative = {0: 'Hôm nay', 1: 'Ngày mai', 2: 'Ngày kia'}.get((target_date.date() - today).days) if today else None
    return f'{relative}, {text}' if relative else text


def bulletin_row(city, origin, h, code, proba_max, tmax, tmin, rain, threshold):
    """Một dòng bản tin. Nhãn và lượng mưa đến từ hai mô hình riêng nên có thể không khớp: ghi chú khi đó."""
    rain = max(0.0, float(rain))
    note = ''
    if code == 2 and rain < 1.0:
        note = 'Nhãn Mưa nhưng lượng mưa dự báo < 1 mm (hai mô hình dự báo riêng)'
    elif code != 2 and rain >= 1.0:
        note = 'Lượng mưa dự báo >= 1 mm nhưng nhãn không phải Mưa (hai mô hình dự báo riêng)'
    return {'city': city, 'origin_date': origin, 'h': h, 'target_date': origin + pd.Timedelta(days=h),
            'weather_code': int(code), 'weather_vi': WEATHER_VI[int(code)], 'tmax': round(float(tmax), 1),
            'tmin': round(float(tmin), 1), 'rain_mm': round(rain, 1),
            'confidence': 'cao' if proba_max >= threshold else 'thấp', 'probability': round(float(proba_max), 3),
            'note': note}


def list_runs(runs_dir=ROOT / 'runs'):
    """Các lần chạy train.py đã có mô hình cuối (models/metadata.json), mới nhất trước."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    return sorted((p for p in runs_dir.iterdir() if (p / 'models' / 'metadata.json').is_file()),
                  key=lambda p: p.name, reverse=True)


def read_metadata(run_dir):
    path = Path(run_dir) / 'models' / 'metadata.json'
    if not path.exists():
        raise ValueError(f'Không thấy {path}; chạy python train.py trước hoặc kiểm tra --run')
    return json.loads(path.read_text(encoding='utf-8'))


def pick_run(runs, target, data_file):
    """Lần chạy mới nhất có mô hình `target` học từ `data_file`.
    Metadata cũ không ghi targets/data_file thì coi là bản đủ 4 đại lượng học từ weather.csv."""
    for run in runs:
        meta = read_metadata(run)
        if target in meta.get('targets', TARGETS) and meta.get('data_file', 'weather.csv') == data_file:
            return run
    return None


TARGET_NAMES = {'weather': 'loại thời tiết', 'tmax': 'nhiệt độ cao', 'tmin': 'nhiệt độ thấp', 'rain': 'lượng mưa'}
SOURCE_NAMES = {'weather.csv': 'Kaggle', 'open_meteo.csv': 'Open-Meteo'}


def runs_with(runs, target):
    """Các lần chạy có mô hình `target`, giữ thứ tự mới nhất trước."""
    return [r for r in runs if target in read_metadata(r).get('targets', TARGETS)]


def describe_run(run, runs=()):
    """Tên cho danh sách chọn: nguồn dữ liệu ('Kaggle'). Nếu trong `runs` có nhiều lần chạy cùng nguồn
    thì thêm tên lần chạy để phân biệt: 'Kaggle (20261110_011246)'."""
    source_of = lambda r: read_metadata(r).get('data_file', 'weather.csv')  # noqa: E731
    source = source_of(run)
    name = SOURCE_NAMES.get(source, source)
    return f'{name} ({Path(run).name})' if sum(source_of(r) == source for r in runs) > 1 else name


def run_contents(run):
    """Các đại lượng có mô hình trong lần chạy, ví dụ 'nhiệt độ cao, nhiệt độ thấp, lượng mưa'."""
    return ', '.join(TARGET_NAMES[t] for t in read_metadata(run).get('targets', TARGETS))


def forecast(wide, run_dir, numbers_run=None):
    """Dự báo 3 ngày sau ngày cuối cùng của dữ liệu cho mỗi thành phố có số liệu ngày đó.
    Loại thời tiết lấy từ run_dir; nhiệt độ và lượng mưa lấy từ numbers_run nếu có (cách tổng hợp)."""
    models_dir = Path(run_dir) / 'models'
    numbers_dir = Path(numbers_run or run_dir) / 'models'
    meta = read_metadata(run_dir)
    columns = meta['feature_columns']
    numbers_columns = read_metadata(numbers_run)['feature_columns'] if numbers_run else columns
    for folder, targets, what in [(models_dir, ['weather'], 'loại thời tiết'),
                                  (numbers_dir, ['tmax', 'tmin', 'rain'], 'nhiệt độ và lượng mưa')]:
        missing = [f'{t}_h{h}.joblib' for t in targets for h in HORIZONS if not (folder / f'{t}_h{h}.joblib').exists()]
        if missing:
            raise ValueError(f'Lần chạy {folder.parent.name} không có mô hình {what} (thiếu {", ".join(missing)}). '
                             'Hãy chọn lần chạy khác hoặc chạy lại train.py.')
    origin = wide['max'].index.max()
    rows = []
    for h in HORIZONS:
        X = make_features(wide, h)
        X = X[X.index.get_level_values('origin_date') == origin]
        if h == 1:
            missing = [c for c in CITIES if c not in set(X['city'])]
            if missing:
                print(f'Không dự báo được {", ".join(missing)}: thiếu số liệu ngày {origin.date()}', file=sys.stderr)
        if X.empty:
            continue
        weather = joblib.load(models_dir / f'weather_h{h}.joblib')
        pred = {t: joblib.load(numbers_dir / f'{t}_h{h}.joblib').predict(X[numbers_columns])
                for t in ['tmax', 'tmin', 'rain']}
        codes, proba = weather.predict(X[columns]), weather.predict_proba(X[columns]).max(axis=1)
        threshold = meta['confidence_threshold'][str(h)]
        for i, city in enumerate(X['city']):
            rows.append(bulletin_row(city, origin, h, codes[i], proba[i], pred['tmax'][i], pred['tmin'][i],
                                     pred['rain'][i], threshold))
    return pd.DataFrame(rows)


def print_bulletin(result, stream=None):
    stream = stream or sys.stdout
    for city, group in result.groupby('city', sort=False):
        print(f'{city} - dự báo từ ngày {group.origin_date.iloc[0].date()}', file=stream)
        for _, r in group.sort_values('h').iterrows():
            line = (f'  {r.target_date:%d/%m}: {r.weather_vi:<16} {r.tmin:.0f}-{r.tmax:.0f}°C  '
                    f'mưa ~{r.rain_mm:.1f} mm  (độ tin cậy {r.confidence})')
            print(line + (f'  * {r.note}' if r.note else ''), file=stream)
        print(file=stream)


def default_runs(runs, live):
    """(mô hình loại thời tiết, mô hình con số). Thời gian thực: con số lấy từ bản học trên Open-Meteo nếu có."""
    weather_run = pick_run(runs, 'weather', 'weather.csv') or pick_run(runs, 'weather', 'open_meteo.csv')
    numbers_run = pick_run(runs, 'tmax', 'open_meteo.csv') if live else None
    return weather_run, numbers_run


def main(argv=None):
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--live', action='store_true', help='Lấy số liệu mới nhất từ Open-Meteo (cần Internet)')
    parser.add_argument('--run', type=Path, help='Lần chạy train.py cho loại thời tiết (mặc định: tự chọn)')
    parser.add_argument('--numbers-run', type=Path, help='Lần chạy train.py cho nhiệt độ, lượng mưa (mặc định: tự chọn)')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/weather.csv')
    parser.add_argument('--out', type=Path, default=Path('forecast.csv'))
    args = parser.parse_args(argv)
    try:
        weather_run, numbers_run = default_runs(list_runs(), args.live)
        weather_run, numbers_run = args.run or weather_run, args.numbers_run or numbers_run
        if weather_run is None:
            raise ValueError('chưa có mô hình loại thời tiết; chạy python train.py trước')
        if args.live and numbers_run is None:
            print('Chưa có mô hình con số học trên Open-Meteo; nhiệt độ và lượng mưa dùng mô hình Kaggle '
                  '(chạy: python train.py --data data/open_meteo.csv --targets tmax tmin rain)', file=sys.stderr)
        if args.live:
            from open_meteo import live_weather
            wide = live_weather()
        else:
            wide = load_weather(args.data)
        result = forecast(wide, weather_run, numbers_run)
    except (OSError, ValueError) as error:
        print(f'Không thực hiện được: {error}', file=sys.stderr)
        return 2
    if result.empty:
        print('Không có thành phố nào đủ số liệu ở ngày cuối cùng để dự báo', file=sys.stderr)
        return 1
    print_bulletin(result)
    result.to_csv(args.out, index=False, encoding='utf-8-sig', date_format='%Y-%m-%d')
    print(f'Đã lưu: {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
