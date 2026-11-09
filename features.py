"""Đọc CSV thời tiết, tạo nhãn, đặc trưng và chia tập cho bài dự báo 3 ngày."""
import sys

import numpy as np
import pandas as pd

CITIES = ['Ha Noi', 'Ho Chi Minh City', 'Nha Trang']
VARIABLES = ['max', 'min', 'wind', 'rain', 'humidi', 'cloud', 'pressure']
WIND_DIRS = 'N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW'.split()
REQUIRED_COLUMNS = ['province', 'date', 'wind_d'] + VARIABLES


def configure_console(streams=None):
    """In UTF-8; ký tự console không hiển thị được thành \\u..., không làm chương trình dừng.
    Bỏ qua stream không có reconfigure (Spyder, Jupyter, IDLE)."""
    for stream in streams or (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, 'reconfigure', None)
        if callable(reconfigure):
            reconfigure(encoding='utf-8', errors='backslashreplace')


def load_weather(path):
    """path: đường dẫn CSV hoặc đối tượng file. Trả về {tên biến: bảng ngày x thành phố}, lịch ngày liên tục, ngày thiếu là NaN.

    Biến gồm VARIABLES và 'wind_deg' (hướng gió đổi sang độ, N = 0°).
    """
    raw = pd.read_csv(path)  # đường dẫn hoặc file đã mở (trang web tải lên)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f'CSV thiếu cột: {", ".join(missing)}')
    raw = raw[raw.province.isin(CITIES)].copy()  # 'Hanoi' là bản trùng của 'Ha Noi' nên bị loại ở đây
    absent = [c for c in CITIES if c not in set(raw.province)]
    if absent:
        raise ValueError(f'CSV không có dữ liệu của: {", ".join(absent)}')
    raw['date'] = pd.to_datetime(raw['date'], format='%Y-%m-%d', errors='raise')
    if raw.duplicated(['province', 'date']).any():
        raise ValueError('CSV có dòng trùng (thành phố, ngày)')
    unknown = set(raw.wind_d.dropna()) - set(WIND_DIRS)
    if unknown:
        raise ValueError(f'Hướng gió không hợp lệ: {sorted(unknown)}')
    raw['wind_deg'] = raw.wind_d.map({d: i * 22.5 for i, d in enumerate(WIND_DIRS)})
    days = pd.date_range(raw.date.min(), raw.date.max(), freq='D')
    return {v: raw.pivot(index='date', columns='province', values=v).reindex(index=days, columns=CITIES).astype(float)
            for v in VARIABLES + ['wind_deg']}


TARGETS = ['weather', 'tmax', 'tmin', 'rain']
HORIZONS = (1, 2, 3)
RAIN_MM, CLOUD_PCT = 1.0, 50.0  # luật nhãn của đồ án: mưa >= 1 mm; không mưa và mây >= 50% là nhiều mây


def weather_code(rain, cloud):
    """2 = Mưa, 1 = Nhiều mây, 0 = Nắng hoặc ít mây; NaN khi thiếu số liệu."""
    code = np.select([rain >= RAIN_MM, cloud >= CLOUD_PCT], [2.0, 1.0], 0.0)
    return pd.DataFrame(code, index=rain.index, columns=rain.columns).where(rain.notna() & cloud.notna())


def _long(tables):
    """{tên cột: bảng ngày x thành phố} -> bảng dài, index (origin_date, city)."""
    out = pd.concat({name: t.stack(future_stack=True) for name, t in tables.items()}, axis=1)
    out.index.names = ['origin_date', 'city']
    return out


def make_targets(wide, h):
    """Nhãn của ngày t+h (theo lịch) cho mỗi ngày gốc t."""
    code = weather_code(wide['rain'], wide['cloud'])
    return _long({f'weather_h{h}': code.shift(-h), f'tmax_h{h}': wide['max'].shift(-h),
                  f'tmin_h{h}': wide['min'].shift(-h), f'rain_h{h}': wide['rain'].shift(-h)})


def make_features(wide, h):
    """Đặc trưng tại ngày gốc t, chỉ dùng số liệu đến hết ngày t (mục 5.2 của thiết kế)."""
    code = weather_code(wide['rain'], wide['cloud'])
    f = {v: wide[v] for v in VARIABLES}
    f['wd_sin'], f['wd_cos'] = np.sin(np.radians(wide['wind_deg'])), np.cos(np.radians(wide['wind_deg']))
    f['trange'] = wide['max'] - wide['min']
    f['code_today'], f['code_lag1'] = code, code.shift(1)
    for v in ['max', 'min', 'rain', 'humidi', 'cloud', 'pressure']:
        f[f'{v}_lag1'], f[f'{v}_lag2'] = wide[v].shift(1), wide[v].shift(2)
    for v in ['pressure', 'cloud', 'humidi', 'max', 'min']:
        f[f'{v}_d1'] = wide[v] - wide[v].shift(1)
    for v in ['max', 'min', 'rain', 'humidi', 'cloud', 'pressure', 'wind']:
        for w in (3, 7):
            f[f'{v}_r{w}'] = wide[v].rolling(w, min_periods=1).mean()
    f['rain_days7'] = (wide['rain'] >= RAIN_MM).astype(float).where(wide['rain'].notna()).rolling(7, min_periods=1).sum()
    out = _long(f)
    out = out[out['max'].notna()]  # bỏ ngày gốc không có số liệu
    target_day = out.index.get_level_values('origin_date') + pd.Timedelta(days=h)
    out['doy_sin'] = np.sin(2 * np.pi * target_day.dayofyear / 365.25)
    out['doy_cos'] = np.cos(2 * np.pi * target_day.dayofyear / 365.25)
    out.insert(0, 'city', out.index.get_level_values('city'))
    return out


TRAIN_SHARE, TRAIN_VAL_SHARE, GAP_DAYS = 0.65, 0.80, 3  # 65% Train / 15% Validation / 20% Test số ngày


def split_dates(wide):
    """Mốc cuối Train và cuối Validation: ngày ở 65% và 80% số ngày có số liệu.
    Với data/weather.csv ra đúng 2017-02-05 và 2018-12-20."""
    days = wide['max'].dropna(how='all').index
    return days[int(len(days) * TRAIN_SHARE) - 1], days[int(len(days) * TRAIN_VAL_SHARE) - 1]


def assign_split(origin_dates, train_end, val_end):
    """Gán 'train' / 'validation' / 'test' theo ngày gốc; GAP_DAYS ngày cuối của Train và Validation là 'gap'
    để ngày cần dự báo (tối đa t+3) không lọt sang tập sau."""
    d = pd.to_datetime(pd.Series(origin_dates))
    train_end, val_end = pd.Timestamp(train_end), pd.Timestamp(val_end)
    gap = pd.Timedelta(days=GAP_DAYS)
    split = np.select([d <= train_end - gap, d <= train_end, d <= val_end - gap, d <= val_end],
                      ['train', 'gap', 'validation', 'gap'], 'test')
    return pd.Series(split, index=pd.Series(origin_dates).index)


def build_dataset(wide, h):
    """(X, targets, split) cho ngày dự báo h; bỏ dòng thiếu nhãn và dòng thuộc khoảng đệm."""
    X = make_features(wide, h)
    targets = make_targets(wide, h).reindex(X.index)
    split = pd.Series(assign_split(X.index.get_level_values('origin_date'), *split_dates(wide)).to_numpy(),
                      index=X.index, name='split')
    keep = targets.notna().all(axis=1) & (split != 'gap')
    return X[keep], targets[keep], split[keep]
