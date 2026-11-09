"""Trang web dự báo thời tiết 3 ngày cho Hà Nội, TP.HCM, Nha Trang.

Chạy: streamlit run app.py
Mặc định dự báo theo thời gian thực: lấy số liệu Open-Meteo đến hết hôm qua (cần Internet), rồi dự báo
hôm nay, ngày mai, ngày kia. Loại thời tiết từ mô hình học trên data/weather.csv (Kaggle); nhiệt độ và
lượng mưa từ mô hình học trên data/open_meteo.csv.
"""
import contextlib
from datetime import datetime
import io

import pandas as pd
import streamlit as st

from features import CITIES, load_weather
from open_meteo import VIETNAM, live_weather
from predict import ROOT, day_label, default_runs, describe_run, forecast, list_runs, run_contents, runs_with

ICON = {0: '☀️', 1: '☁️', 2: '🌧️'}
CITY_VI = {'Ha Noi': 'Hà Nội', 'Ho Chi Minh City': 'TP. Hồ Chí Minh', 'Nha Trang': 'Nha Trang'}
TARGET_VI = {'weather': 'Loại thời tiết (accuracy)', 'tmax': 'Nhiệt độ cao (sai số °C)',
             'tmin': 'Nhiệt độ thấp (sai số °C)', 'rain': 'Lượng mưa (sai số mm)'}
LIVE, FROM_CSV = 'Thời gian thực (Open-Meteo)', 'Từ file CSV'

st.set_page_config(page_title='Dự báo thời tiết', page_icon='🌦️', layout='wide')


@st.cache_data(ttl=3600, show_spinner='Đang lấy số liệu mới nhất từ Open-Meteo...')
def fetch_live():
    """Số liệu đến hết hôm qua; lưu tạm 1 giờ để không gọi Open-Meteo quá nhiều lần."""
    return live_weather(), datetime.now(VIETNAM)


@st.cache_data(show_spinner='Đang đọc file...')
def read_csv(csv_bytes):
    return load_weather(io.BytesIO(csv_bytes) if csv_bytes is not None else ROOT / 'data' / 'weather.csv')


def run_forecast(wide, weather_run, numbers_run):
    warnings = io.StringIO()
    with contextlib.redirect_stderr(warnings):
        result = forecast(wide, weather_run, numbers_run)
    return result, warnings.getvalue().strip()


def metric_rows(run_dir, targets):
    """Kết quả trên Test của một lần chạy, gộp 3 thành phố: 'RF (baseline tốt nhất)' cho từng ngày."""
    if run_dir is None or not (run_dir / 'results' / 'test_metrics.csv').exists():
        return pd.DataFrame()
    path = run_dir / 'results' / 'test_metrics.csv'
    m = pd.read_csv(path)
    m = m[(m.city == 'Tất cả') & m.target.isin(targets)].copy()
    is_acc = m.metric == 'accuracy'
    best = m[['persistence', 'monthly']].max(axis=1).where(is_acc, m[['persistence', 'monthly']].min(axis=1))
    fmt = lambda v, acc: f'{100 * v:.1f}%' if acc else f'{v:.2f}'  # noqa: E731
    m['cell'] = [f'{fmt(r, a)} (baseline {fmt(b, a)})' for r, b, a in zip(m.random_forest, best, is_acc)]
    return m.pivot(index='target', columns='h', values='cell')


def live_weather_row(weather_run):
    """Accuracy loại thời tiết khi đầu vào là số liệu Open-Meteo (evaluate_live.py)."""
    path = weather_run / 'results' / 'live_weather_accuracy.csv'
    if not path.exists():
        return pd.DataFrame()
    a = pd.read_csv(path).set_index('h')
    cells = {h: f'{100 * r.accuracy_open_meteo_input:.1f}% (đầu vào Kaggle: {100 * r.accuracy_kaggle_input:.1f}%)'
             for h, r in a.iterrows()}
    return pd.DataFrame([cells], index=['weather'])


def accuracy_table(weather_run, numbers_run, live):
    if live:
        weather = live_weather_row(weather_run)
        numbers = metric_rows(numbers_run or weather_run, ['tmax', 'tmin', 'rain'])
        table = pd.concat([weather, numbers])
    else:
        table = metric_rows(weather_run, list(TARGET_VI))
    if table.empty:
        return table
    table = table.reindex([t for t in TARGET_VI if t in table.index])
    table.index = [TARGET_VI[t] for t in table.index]
    table.columns = [f'Ngày +{h}' for h in table.columns]
    return table


st.title('🌦️ Dự báo thời tiết')
st.caption('Hà Nội · TP. Hồ Chí Minh · Nha Trang. Dự báo bằng mô hình máy học Random Forest.')

runs = list_runs()
if not runs:
    st.error('Chưa có mô hình. Mở terminal trong thư mục `forecast_3days` và chạy `python train.py`, '
             'rồi tải lại trang này. Xem README để biết các bước.')
    st.stop()

with st.sidebar:
    st.header('Thiết lập')
    mode = st.radio('Nguồn số liệu', [LIVE, FROM_CSV])
    live = mode == LIVE
    uploaded = None
    if not live:
        uploaded = st.file_uploader(
            'Tải lên file CSV (tuỳ chọn)', type='csv',
            help='Cùng định dạng với data/weather.csv. Dự báo bắt đầu từ ngày cuối cùng trong file. '
                 'Không tải thì dùng data/weather.csv có sẵn.')
    chosen = st.multiselect('Thành phố', CITIES, default=CITIES, format_func=CITY_VI.get)
    weather_run, numbers_run = default_runs(runs, live)
    with st.expander('Mô hình đang dùng'):
        weather_options = runs_with(runs, 'weather')
        weather_run = st.selectbox('Loại thời tiết', weather_options,
                                   format_func=lambda p: describe_run(p, weather_options),
                                   index=weather_options.index(weather_run) if weather_run in weather_options else 0)
        st.caption(f'Gồm: {run_contents(weather_run)}')
        if live:
            options = [None] + runs_with(runs, 'tmax')
            numbers_run = st.selectbox(
                'Nhiệt độ, lượng mưa', options, index=options.index(numbers_run) if numbers_run in options else 0,
                format_func=lambda p: 'Dùng chung mô hình trên' if p is None else describe_run(p, options[1:]))
            if numbers_run is not None:
                st.caption(f'Gồm: {run_contents(numbers_run)}')
    if live and st.button('Cập nhật số liệu'):
        fetch_live.clear()

try:
    if live:
        wide, fetched_at = fetch_live()
    else:
        wide, fetched_at = read_csv(uploaded.getvalue() if uploaded else None), None
    result, warning = run_forecast(wide, weather_run, numbers_run if live else None)
except (ValueError, KeyError) as error:
    st.error(f'Không lấy được số liệu: {error}')
    st.stop()

if warning:
    st.warning(warning)
if result.empty:
    st.error('Không có thành phố nào đủ số liệu ở ngày cuối cùng để dự báo.')
    st.stop()

origin = result.origin_date.iloc[0]
today = fetched_at.date() if live else None
if live:
    st.subheader(f'Dự báo hôm nay, {day_label(pd.Timestamp(today))}/{today:%Y}')
    st.caption(f'Số liệu Open-Meteo đến hết ngày {origin:%d/%m/%Y}, lấy lúc {fetched_at:%H:%M} (giờ Việt Nam). '
               'Tự cập nhật sau mỗi giờ.')
    if live and numbers_run is None:
        st.info('Chưa có mô hình nhiệt độ, lượng mưa học trên Open-Meteo nên đang dùng mô hình Kaggle. '
                'Chạy `python train.py --data data/open_meteo.csv --targets tmax tmin rain` để có số chính xác hơn.')
else:
    st.subheader(f'Dự báo từ ngày {origin:%d/%m/%Y} (ngày cuối cùng trong file)')

shown = [c for c in chosen if c in set(result.city)]
for column, city in zip(st.columns(max(len(shown), 1)), shown):
    with column:
        st.markdown(f'### {CITY_VI[city]}')
        for _, r in result[result.city == city].sort_values('h').iterrows():
            with st.container(border=True):
                st.markdown(f'**{day_label(r.target_date, today)}**')
                st.markdown(f'#### {ICON[r.weather_code]} {r.weather_vi}')
                st.markdown(f'🌡️ **{r.tmin:.0f}–{r.tmax:.0f}°C** &nbsp;&nbsp; 💧 **~{r.rain_mm:.1f} mm**')
                st.caption(f'Độ tin cậy: {r.confidence} (xác suất {r.probability:.0%})')
                if r.note:
                    st.caption(f'⚠️ {r.note}')

with st.expander('Dự báo này chính xác đến đâu?'):
    table = accuracy_table(weather_run, numbers_run, live)
    if table.empty:
        st.write('Chưa có kết quả đánh giá cho mô hình này.')
    else:
        st.table(table)
    if live:
        st.caption('Loại thời tiết: mô hình học từ dữ liệu Kaggle, đo khi nhận số liệu Open-Meteo trên giai đoạn '
                   '12/2018–06/2021 (chạy evaluate_live.py). Nhiệt độ, lượng mưa: mô hình học từ Open-Meteo, đo trên '
                   'giai đoạn Test của nó (20% số ngày cuối, 04/2023–11/2026). Baseline: cách đoán đơn giản tốt hơn trong hai cách "giống hôm nay" và '
                   '"theo tháng".')
    else:
        st.caption('Kết quả trên tập Test. Baseline: cách đoán đơn giản tốt hơn trong hai cách "giống hôm nay" '
                   'và "theo tháng".')
    st.markdown('- Đây là dự báo bằng máy học từ số liệu theo ngày. Các ứng dụng thời tiết chuyên nghiệp dùng mô hình '
                'khí quyển nên chính xác hơn.\n'
                '- Nhãn và các con số được dự báo bằng các mô hình riêng nên đôi khi không khớp nhau.\n'
                '- "Độ tin cậy" là xác suất lớn nhất của Random Forest, chưa được hiệu chỉnh.')

with st.expander('Bảng chi tiết'):
    st.dataframe(result, hide_index=True)
    st.download_button('Tải bảng dự báo (CSV)', result.to_csv(index=False).encode('utf-8-sig'),
                       file_name=f'du_bao_{origin:%Y%m%d}.csv', mime='text/csv')
