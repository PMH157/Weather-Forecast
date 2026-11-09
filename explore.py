"""Mô tả dữ liệu 3 thành phố: số ngày, ngày thiếu, thống kê từng biến, tỷ lệ Nắng/Mây/Mưa theo tháng.

Chạy: python explore.py [--data data/weather.csv] [--out outputs/explore]
"""
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from features import CITIES, VARIABLES, configure_console, load_weather, weather_code  # noqa: E402

ROOT = Path(__file__).resolve().parent
LABELS = ['Nắng hoặc ít mây', 'Nhiều mây', 'Mưa']


def main(argv=None):
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/weather.csv')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/explore')
    args = parser.parse_args(argv)
    wide = load_weather(args.data)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for city in CITIES:
        present = wide['max'][city].notna()
        row = {'city': city, 'first_day': present.idxmax().date(), 'last_day': present[::-1].idxmax().date(),
               'days_with_data': int(present.sum()), 'missing_days': int((~present).sum())}
        for v in VARIABLES:
            s = wide[v][city]
            row.update({f'{v}_min': s.min(), f'{v}_mean': round(s.mean(), 2), f'{v}_max': s.max()})
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out / 'data_summary.csv', index=False, encoding='utf-8-sig')

    code = weather_code(wide['rain'], wide['cloud'])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    shares = {}
    for ax, city in zip(axes, CITIES):
        c = code[city].dropna()
        share = pd.crosstab(c.index.month, c, normalize='index').reindex(columns=[0.0, 1.0, 2.0], fill_value=0)
        share.columns = LABELS
        shares[city] = share
        share.plot.bar(stacked=True, ax=ax, color=['#f2b134', '#9aa5b1', '#3b7dd8'], width=0.85, legend=False)
        ax.set_title(city)
        ax.set_xlabel('Tháng')
    axes[0].set_ylabel('Tỷ lệ số ngày')
    axes[-1].legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(args.out / 'class_share_by_city_month.png', dpi=150)
    pd.concat(shares, names=['city', 'month']).round(3).to_csv(args.out / 'class_share_by_city_month.csv', encoding='utf-8-sig')

    overall = pd.DataFrame({city: code[city].value_counts(normalize=True).reindex([0.0, 1.0, 2.0]).values
                            for city in CITIES}, index=LABELS)
    print(summary[['city', 'first_day', 'last_day', 'days_with_data', 'missing_days']].to_string(index=False))
    print('\nTỷ lệ các lớp thời tiết (cả giai đoạn):')
    print((overall * 100).round(1).to_string())
    print(f'\nĐã lưu tại: {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
