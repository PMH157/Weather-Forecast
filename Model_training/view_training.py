"""Mở biểu đồ đã lưu trong trình duyệt; tự tìm lần chạy, không train mô hình."""
import argparse
import csv
from html import escape
from pathlib import Path
import sys
from urllib.parse import quote
import webbrowser

from modeling import configure_console

ROOT = Path(__file__).resolve().parent


def figure_paths(run_dir):
    directory = Path(run_dir) / 'results/visualizations'
    index = directory / 'curves_index.csv'
    if index.is_file():
        with index.open(encoding='utf-8-sig', newline='') as handle:
            names = [Path(row.get('figure', '')).name for row in csv.DictReader(handle)
                     if row.get('status') == 'ok' and row.get('figure')]
        return list(dict.fromkeys(directory / name for name in names
                                 if name.lower().endswith('.png') and (directory / name).is_file()))
    return sorted(directory.glob('*.png'))


def describe_runs(runs_dir):
    directory = Path(runs_dir)
    if not directory.is_dir():
        return []
    rows = []
    for run in directory.iterdir():
        if not run.is_dir():
            continue
        images = figure_paths(run)
        rows.append({'run_dir': run, 'model_saved': (run / 'models/final_pipeline.joblib').is_file(),
                     'selection_saved': (run / 'models/selection.json').is_file(),
                     'figures': len(images),
                     'updated': max((p.stat().st_mtime_ns for p in images), default=run.stat().st_mtime_ns)})
    return sorted(rows, key=lambda row: (row['updated'], row['run_dir'].name), reverse=True)


def latest_run_with_figures(runs_dir):
    for row in describe_runs(runs_dir):
        if row['figures']:
            return row['run_dir']
    raise FileNotFoundError(
        f'Chưa tìm thấy biểu đồ trong {runs_dir}. '
        'Dùng python view_training.py --list-runs để xem trạng thái các lần chạy.')


def create_gallery(run_dir):
    run_dir = Path(run_dir).resolve()
    images = figure_paths(run_dir)
    if not images:
        raise FileNotFoundError(f'Lần chạy {run_dir.name} chưa có biểu đồ PNG thành công.')
    sections, navigation = [], []
    for number, image in enumerate(images, start=1):
        title = image.stem.replace('_', ' ').title()
        navigation.append(f'<a href="#curve-{number}">{escape(title)}</a>')
        csv_path = image.with_suffix('.csv')
        links = f'<a href="{escape(quote(image.name), quote=True)}" target="_blank" rel="noopener">Mở ảnh gốc</a>'
        if csv_path.is_file():
            links += f'<a href="{escape(quote(csv_path.name), quote=True)}" download>Tải số liệu CSV</a>'
        sections.append(
            f'<section id="curve-{number}"><div class="heading"><h2>{escape(title)}</h2>'
            f'<div class="links">{links}</div></div>'
            f'<img src="{escape(quote(image.name), quote=True)}" alt="{escape(title, quote=True)}" loading="lazy"></section>')
    model_status = ('Đã có final_pipeline.joblib' if (run_dir / 'models/final_pipeline.joblib').is_file()
                    else 'Chưa có final_pipeline.joblib trong lần chạy này')
    document = '''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Biểu đồ huấn luyện</title><style>
*{box-sizing:border-box}body{margin:0;background:#f1f5f9;color:#0f172a;font:16px/1.5 system-ui,Arial,sans-serif}
main{max-width:1280px;margin:auto;padding:28px 20px}h1{margin:0 0 10px;font-size:30px}h2{margin:0;font-size:20px}
p{margin:8px 0}.status{color:#475569}.path{overflow-wrap:anywhere;font-size:14px}
nav{display:flex;gap:8px;flex-wrap:wrap;margin:22px 0}a{color:#1d4ed8}nav a{background:white;padding:6px 10px;border:1px solid #cbd5e1;border-radius:8px;text-decoration:none;font-size:14px}
section{background:white;margin:20px 0;border:1px solid #cbd5e1;border-radius:12px;overflow:hidden;scroll-margin-top:16px}
.heading{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;padding:18px 20px;border-bottom:1px solid #e2e8f0}
.links{display:flex;gap:16px;font-size:14px;align-items:center}img{display:block;width:100%;height:auto}
</style></head><body><main>'''
    document += (f'<h1>Biểu đồ quá trình huấn luyện</h1><p>Lần chạy: <strong>{escape(run_dir.name)}</strong></p>'
                 f'<p class="status">{len(images)} biểu đồ · {escape(model_status)}</p>'
                 '<p>Xanh: Train · Cam: Validation. Chọn tên biểu đồ để chuyển đến hình tương ứng.</p>'
                 '<p class="status">Trang này đọc kết quả đã lưu; mở lại trang không huấn luyện mô hình.</p>'
                 f'<p class="path">{escape(str(run_dir))}</p><nav>{"".join(navigation)}</nav>'
                 f'{"".join(sections)}</main></body></html>')
    destination = run_dir / 'results/visualizations/training_curves.html'
    destination.write_text(document, encoding='utf-8')
    return destination


def open_training_gallery(run_dir=None, *, runs_dir=None, open_browser=True):
    configure_console()
    selected = Path(run_dir) if run_dir is not None else latest_run_with_figures(runs_dir or ROOT / 'runs')
    gallery = create_gallery(selected)
    print(f'Trang biểu đồ: {gallery}', flush=True)
    if open_browser:
        try:
            opened = webbrowser.open(gallery.as_uri(), new=2)
        except (OSError, webbrowser.Error):
            opened = False
        if not opened:
            print('Không tự mở được trình duyệt. Mở file training_curves.html tại đường dẫn trên.', flush=True)
    return gallery


def main(argv=None):
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, help='Tùy chọn: một lần chạy cụ thể')
    parser.add_argument('--runs-dir', type=Path, default=ROOT / 'runs', help='Nơi tự tìm các lần chạy')
    parser.add_argument('--list-runs', action='store_true', help='Liệt kê trạng thái mô hình và số biểu đồ')
    parser.add_argument('--no-open', action='store_true', help='Tạo trang HTML, không tự mở trình duyệt')
    args = parser.parse_args(argv)
    try:
        if args.list_runs:
            rows = describe_runs(args.runs_dir)
            if not rows:
                print(f'Chưa có thư mục lần chạy trong {args.runs_dir}')
            for row in rows:
                status = 'đã lưu' if row['model_saved'] else 'chưa lưu'
                print(f'{row["run_dir"].name} | Mô hình: {status} | Biểu đồ: {row["figures"]}')
            return 0
        open_training_gallery(args.run_dir, runs_dir=args.runs_dir, open_browser=not args.no_open)
        return 0
    except (OSError, ValueError) as error:
        print(f'Không mở được biểu đồ: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
