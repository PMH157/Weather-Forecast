"""Lấy số liệu thời tiết theo ngày từ Open-Meteo (miễn phí, không cần key) cho Hà Nội, TP.HCM, Nha Trang.

Tải lịch sử để train:  python open_meteo.py [--start 2009-01-01] [--end <hôm qua>] [--out data/open_meteo.csv]
Trang web dùng live_weather() để lấy các ngày gần nhất đến hết hôm qua.
Số liệu được đổi sang đúng định dạng cột của data/weather.csv để dùng chung features.py.
"""
import argparse
from datetime import date, datetime, timedelta, timezone
import io
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from features import CITIES, WIND_DIRS, configure_console, load_weather

ROOT = Path(__file__).resolve().parent
COORDS = {'Ha Noi': (21.0285, 105.8542), 'Ho Chi Minh City': (10.8231, 106.6297), 'Nha Trang': (12.2388, 109.1967)}
ARCHIVE_URL = 'https://archive-api.open-meteo.com/v1/archive'
FORECAST_URL = 'https://api.open-meteo.com/v1/forecast'
# Biến Open-Meteo -> cột của data/weather.csv (gió trung bình khớp thang của Kaggle hơn gió lớn nhất)
COLUMNS = {'temperature_2m_max': 'max', 'temperature_2m_min': 'min', 'wind_speed_10m_mean': 'wind',
           'precipitation_sum': 'rain', 'relative_humidity_2m_mean': 'humidi', 'cloud_cover_mean': 'cloud',
           'pressure_msl_mean': 'pressure'}
WIND_DIRECTION = 'wind_direction_10m_dominant'
VIETNAM = timezone(timedelta(hours=7))


def wind_label(degrees):
    """Độ (0–360) -> một trong 16 hướng như cột wind_d của CSV."""
    if degrees is None or pd.isna(degrees):
        return None
    return WIND_DIRS[int(round(float(degrees) / 22.5)) % 16]


def vietnam_yesterday(now=None):
    """Ngày hôm qua theo giờ Việt Nam: ngày cuối cùng có số liệu trọn ngày."""
    now = now or datetime.now(timezone.utc)
    return now.astimezone(VIETNAM).date() - timedelta(days=1)


def build_url(base, **params):
    query = {'latitude': ','.join(str(COORDS[c][0]) for c in CITIES),
             'longitude': ','.join(str(COORDS[c][1]) for c in CITIES),
             'daily': ','.join(list(COLUMNS) + [WIND_DIRECTION]), 'timezone': 'Asia/Ho_Chi_Minh', **params}
    return base + '?' + urllib.parse.urlencode(query)


def fetch_json(url, opener=urllib.request.urlopen, sleep=time.sleep, retries=5, timeout=60):
    """Gọi Open-Meteo; bị giới hạn lượt gọi (HTTP 429), lỗi máy chủ hay mất mạng thì chờ rồi thử lại."""
    last = ''
    for attempt in range(retries):
        try:
            with opener(url, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code != 429 and error.code < 500:
                raise ValueError(f'Open-Meteo từ chối yêu cầu (HTTP {error.code})') from None
            last = f'HTTP {error.code}'
        except (urllib.error.URLError, TimeoutError, OSError):
            last = 'không kết nối được'
        if attempt < retries - 1:
            sleep(5 * 2 ** attempt)
    if last == 'HTTP 429':
        raise ValueError(f'Open-Meteo đang giới hạn số lần gọi từ máy này (đã thử {retries} lần). Thử lại sau vài phút.')
    raise ValueError(f'Không lấy được số liệu từ Open-Meteo ({last}) sau {retries} lần thử. '
                     'Kiểm tra kết nối Internet rồi thử lại sau ít phút.')


def to_weather_frame(blocks, until=None):
    """Phản hồi Open-Meteo (mỗi thành phố một khối, theo thứ tự CITIES) -> bảng cùng cột với data/weather.csv.
    until: bỏ các ngày sau ngày này (ví dụ ngày hôm nay chưa trọn)."""
    if isinstance(blocks, dict):
        blocks = [blocks]
    if len(blocks) != len(CITIES):
        raise ValueError(f'Open-Meteo trả về {len(blocks)} địa điểm, cần {len(CITIES)}')
    frames = []
    for city, block in zip(CITIES, blocks):
        daily = block['daily']
        frame = pd.DataFrame({'province': city, 'date': daily['time']})
        for source, column in COLUMNS.items():
            frame[column] = pd.to_numeric(pd.Series(daily[source], dtype='object'), errors='coerce')
        frame['wind_d'] = [wind_label(d) for d in daily[WIND_DIRECTION]]
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    if until is not None:
        out = out[out.date <= until.isoformat()]
    return out[['province', 'max', 'min', 'wind', 'wind_d', 'rain', 'humidi', 'cloud', 'pressure', 'date']]


def fetch_history(start, end):
    return to_weather_frame(fetch_json(build_url(ARCHIVE_URL, start_date=start, end_date=end)))


def fetch_recent(days=10, now=None):
    """Các ngày gần nhất đến hết hôm qua (giờ Việt Nam)."""
    blocks = fetch_json(build_url(FORECAST_URL, past_days=days, forecast_days=1))
    return to_weather_frame(blocks, until=vietnam_yesterday(now))


def live_weather(now=None):
    """Số liệu gần nhất dưới dạng bảng ngày x thành phố, sẵn sàng cho predict.forecast()."""
    return load_weather(io.StringIO(fetch_recent(now=now).to_csv(index=False)))


def main(argv=None):
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default='2009-01-01')
    parser.add_argument('--end', default=vietnam_yesterday().isoformat())
    parser.add_argument('--out', type=Path, default=ROOT / 'data' / 'open_meteo.csv')
    args = parser.parse_args(argv)
    try:
        frame = fetch_history(args.start, args.end)
    except ValueError as error:
        print(f'Không thực hiện được: {error}', file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.groupby('province').date.agg(['count', 'min', 'max']).to_string())
    print(f'Đã lưu: {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
