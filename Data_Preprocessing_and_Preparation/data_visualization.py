"""Xuất bảng, biểu đồ EDA và mở báo cáo HTML; không huấn luyện mô hình."""
import argparse
from copy import deepcopy
import html
import json
from pathlib import Path
import sys
import webbrowser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocessing import (DEFAULT_CONFIG, LABEL_VI, clean_daily,
                           build_supervised, split_by_time)

ROOT = Path(__file__).resolve().parent
plt.rcParams['font.family'] = 'DejaVu Sans'


def generate(raw_path, output, config_path=None, scope='train'):
    cfg = deepcopy(DEFAULT_CONFIG)
    if config_path and config_path.exists():
        cfg.update(json.loads(config_path.read_text(encoding='utf-8-sig')))
    raw = pd.read_csv(raw_path, encoding='utf-8-sig')
    daily, audit, issues, _ = clean_daily(raw, cfg)
    dataset, _ = build_supervised(daily, cfg)
    _, splits, _ = split_by_time(dataset, cfg)
    # Cắt theo ngày nguồn của Train; không dùng dữ liệu tương lai để khám phá mặc định.
    end = splits['train'].origin_date.max() if scope == 'train' else daily.date.max()
    start = splits['train'].origin_date.min() if scope == 'train' else daily.date.min()
    calendar = daily[daily.date.between(start, end)].copy()
    observed = calendar[calendar.source_row_present].copy()
    if observed.empty:
        raise ValueError('Không có dữ liệu trong phạm vi được chọn.')
    output.mkdir(parents=True, exist_ok=True)
    entries = []

    def save(stem, title, frame, note, figure):
        figure.savefig(output / (stem + '.png'), dpi=160, bbox_inches='tight')
        plt.close(figure)
        frame.to_csv(output / (stem + '.csv'), index=False, encoding='utf-8-sig')
        entries.append((stem, title, note))

    def table(stem, title, frame, note):
        display = frame.copy().fillna('Thiếu')
        for col in display:
            display[col] = display[col].map(lambda v: str(round(v, 3)) if isinstance(v, float) else str(v))
        fig, ax = plt.subplots(figsize=(max(10, len(frame.columns)*1.6), max(2.5, len(frame)*0.4+1.4)))
        ax.axis('off')
        ax.set_title(title, pad=18, fontweight='bold')
        grid = ax.table(cellText=display.values, colLabels=display.columns, loc='center', cellLoc='center')
        grid.auto_set_font_size(False)
        grid.set_fontsize(8)
        grid.scale(1, 1.6)
        for (row, _), cell in grid.get_celld().items():
            cell.set_facecolor('#dceaf7' if row == 0 else ('#f3f6fa' if row % 2 else 'white'))
        save(stem, title, frame, note, fig)

    table('01_raw', 'Mẫu dữ liệu thô — 10 dòng đầu', raw.head(10),
          'Dữ liệu đầu vào trước xử lý. Bảng minh họa lấy 10 dòng đầu của file gốc; không phải bảng đã chuẩn hóa.')
    cols = ['city', 'date', 'temperature_max_c', 'temperature_min_c', 'rain_mm',
            'humidity_pct', 'cloud_cover_pct', 'wind_speed_m_s', 'weather_today']
    preview = observed[cols].head(10).copy()
    preview['date'] = preview.date.dt.strftime('%Y-%m-%d')
    table('02_clean', 'Mẫu dữ liệu sau làm sạch', preview,
          'Nhiệt độ: °C; mưa: mm/ngày theo quy ước nguồn; gió: m/s; độ ẩm và mây: %. '
          'Giá trị ngoài ngưỡng chuyển thành thiếu. Đây là bảng làm sạch, chưa điền trung vị hoặc scale; bước đó chỉ học từ Train.')
    numeric = ['temperature_max_c', 'temperature_min_c', 'rain_mm', 'humidity_pct', 'cloud_cover_pct', 'wind_speed_m_s']
    stats = observed[numeric].describe().round(3).rename_axis('Thống kê').reset_index()
    table('03_statistics', 'Thống kê các biến số', stats, 'count là số giá trị có dữ liệu; std là độ lệch chuẩn.')
    missing = calendar[numeric].isna().sum().rename('Số ô thiếu').rename_axis('Biến').reset_index()
    missing['Tỷ lệ (%)'] = (missing['Số ô thiếu'] / len(calendar) * 100).round(2)
    table('04_missing', 'Giá trị thiếu sau làm sạch', missing,
          'Bao gồm các ngày được bổ sung vào lịch và các ô ngoài ngưỡng; không tự điền giá trị để vẽ biểu đồ.')
    counts = observed.weather_today.value_counts().reindex(list(LABEL_VI), fill_value=0)
    dist = pd.DataFrame({'Thời tiết': [LABEL_VI[k] for k in counts.index], 'Số ngày': counts.values})
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(dist['Thời tiết'], dist['Số ngày'], color=['#eba938', '#7394b2', '#3182bd'])
    for i, v in enumerate(counts.values):
        ax.text(v, i, ' ' + str(v), va='center')
    ax.set(xlabel='Số bản ghi thành phố–ngày', title='Phân bố nhãn thời tiết của ngày quan trắc')
    save('05_weather_distribution', 'Phân bố thời tiết', dist,
         f'Nhãn suy ra từ ngưỡng mưa {cfg["rain_threshold_mm"]} mm và mây {cfg["cloud_threshold_pct"]}%; '
         f'không phải mô tả Weather có sẵn từ nguồn. Có {observed.weather_today.isna().sum()} ngày chưa đủ dữ liệu để gán nhãn. '
         'Đây là nhãn ngày hiện tại, không phải phân bố target ngày mai.', fig)
    month = observed.assign(month=observed.date.dt.month)
    temp = month.groupby(['city', 'month'])[['temperature_max_c', 'temperature_min_c']].mean().reset_index()
    fig, axes = plt.subplots(len(temp.city.unique()), 1, figsize=(10, 3.4*len(temp.city.unique())), squeeze=False)
    for ax, (city, part) in zip(axes.flat, temp.groupby('city')):
        part = part.set_index('month').reindex(range(1, 13))
        ax.plot(part.index, part.temperature_max_c, 'o-', label='Trung bình nhiệt độ cao nhất ngày')
        ax.plot(part.index, part.temperature_min_c, 'o-', label='Trung bình nhiệt độ thấp nhất ngày')
        ax.set(title=city, xlabel='Tháng', ylabel='°C', xticks=range(1, 13))
        ax.legend(fontsize=8)
    fig.tight_layout()
    save('06_monthly_temperature', 'Nhiệt độ theo tháng và thành phố', temp,
         'Trung bình riêng nhiệt độ cao nhất/thấp nhất ngày qua các năm trong phạm vi chọn; không phải nhiệt độ trung bình ngày.', fig)
    calendar['year'] = calendar.date.dt.year
    calendar['month'] = calendar.date.dt.month
    rain = calendar.groupby(['city', 'year', 'month']).agg(
        total_mm=('rain_mm', lambda s: s.sum(min_count=1)), valid_days=('rain_mm', 'count'),
        calendar_days=('date', 'size')).reset_index()
    rain['expected_days'] = pd.to_datetime(dict(year=rain.year, month=rain.month, day=1)).dt.days_in_month
    rain['complete'] = rain.valid_days.eq(rain.expected_days) & rain.calendar_days.eq(rain.expected_days)
    rain.to_csv(output / 'monthly_rain_completeness.csv', index=False, encoding='utf-8-sig')
    avg = rain[rain.complete].groupby(['city', 'month']).agg(mean_total_mm=('total_mm', 'mean'), complete_months=('year', 'size')).reset_index()
    cities = sorted(observed.city.unique())
    fig, axes = plt.subplots(len(cities), 1, figsize=(10, 3.2*len(cities)), squeeze=False)
    for ax, city in zip(axes.flat, cities):
        part = avg[avg.city.eq(city)].set_index('month').reindex(range(1, 13))
        ax.bar(part.index, part.mean_total_mm, color='#3182bd')
        ax.set(title=city, xlabel='Tháng', ylabel='mm/tháng', xticks=range(1, 13))
        if part.mean_total_mm.notna().sum() == 0:
            ax.text(.5, .5, 'Không có tháng đủ dữ liệu', transform=ax.transAxes, ha='center')
    fig.tight_layout()
    save('07_monthly_rain', 'Trung bình tổng lượng mưa tháng qua các năm', avg,
         'Cộng lượng mưa ngày trong từng tháng–năm rồi lấy trung bình giữa các năm. '
         'Chỉ dùng tháng đủ tất cả ngày và không thiếu rain_mm; không coi ngày thiếu là 0. '
         'CSV monthly_rain_completeness.csv ghi rõ các tháng được giữ/loại.', fig)
    details = {'scope': scope, 'start': str(start.date()), 'end': str(end.date()),
               'observed_rows_in_scope': len(observed), 'audit_full_input': audit,
               'flagged_cells_full_input': len(issues), 'classifier_trained': False}
    (output / 'eda_summary.json').write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding='utf-8')
    cards = ''.join(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(note)}</p>'
                    f'<a href="{stem}.png"><img src="{stem}.png" alt="{html.escape(title)}"></a>'
                    f'<p><a href="{stem}.csv">Tải số liệu CSV</a> · <a href="{stem}.png">Mở ảnh gốc</a></p></section>'
                    for stem, title, note in entries)
    report = output / 'index.html'
    report.write_text('<!doctype html><html lang="vi"><meta charset="utf-8"><title>Khám phá dữ liệu thời tiết</title>'
                      '<style>body{font:16px Arial;max-width:1200px;margin:32px auto;padding:16px;background:#f3f6fa}'
                      'section{background:white;padding:22px;margin:22px 0;border-radius:12px}img{max-width:100%}p{line-height:1.6}</style>'
                      f'<h1>Khám phá dữ liệu thời tiết — Thành viên 2</h1><p>Phạm vi: {scope}; '
                      f'{start.date()} đến {end.date()}. Mặc định dùng giai đoạn Train để hỗ trợ phân tích trước huấn luyện.</p>'
                      '<p>Bảng thô minh họa file gốc; các thống kê và biểu đồ dùng dữ liệu đã làm sạch trong phạm vi trên. '
                      'Không thay đổi dữ liệu đầu vào, không fit mô hình, không học tham số tiền xử lý.</p>' + cards + '</html>', encoding='utf-8')
    return report


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'data/interim/weather_raw.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/member2/eda')
    parser.add_argument('--config', type=Path, default=ROOT/'config/preprocessing.json')
    parser.add_argument('--scope', choices=['train', 'all'], default='train')
    parser.add_argument('--no-open', action='store_true', help='Chỉ xuất ảnh, không mở trình duyệt')
    args = parser.parse_args()
    try:
        report = generate(args.input.resolve(), args.output.resolve(), args.config, args.scope)
        print('Đã xuất bảng và biểu đồ:', report)
        if not args.no_open and not webbrowser.open(report.as_uri(), new=2):
            print('Hãy mở file index.html ở đường dẫn trên để xem.')
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print('Không tạo được báo cáo:', exc, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
