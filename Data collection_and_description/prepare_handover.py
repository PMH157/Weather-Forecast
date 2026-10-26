"""Kiểm tra dữ liệu gốc và xuất phạm vi ba thành phố cho thành viên 2.

Chỉ chọn địa điểm và sắp xếp bảng. Không điền thiếu, mã hóa, tạo nhãn,
chuẩn hóa giá trị, loại ngoại lệ hoặc chia tập huấn luyện.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
EXPECTED_COLUMNS = ["province", "max", "min", "wind", "wind_d", "rain", "humidi", "cloud", "pressure", "date"]
NUMERIC = ["max", "min", "wind", "rain", "humidi", "cloud", "pressure"]
COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def dump_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/weather.csv")
    args = parser.parse_args(argv)
    data_dir, meta = ROOT / "data/interim", ROOT / "reports/member1"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError("Unexpected CSV columns; inspect the source before changing the mapping")
    dates = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    numeric = df[NUMERIC].apply(pd.to_numeric, errors="coerce")
    checks = {
        "invalid_date": dates.isna(),
        "missing_any_column": df.isna().any(axis=1),
        "non_numeric_value": numeric.isna().any(axis=1),
        "max_below_min": numeric["max"] < numeric["min"],
        "negative_rain": numeric["rain"] < 0,
        "negative_wind": numeric["wind"] < 0,
        "humidity_outside_0_100": ~numeric["humidi"].between(0, 100),
        "cloud_outside_0_100": ~numeric["cloud"].between(0, 100),
        "unknown_wind_direction": ~df["wind_d"].isin(COMPASS),
        "duplicate_exact_row_after_first": df.duplicated(),
        "duplicate_province_date_after_first": df.duplicated(["province", "date"]),
    }
    summary = {
        "source_file": args.input.name,
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "rows": len(df), "columns": len(df.columns),
        "distinct_location_strings": int(df["province"].nunique()),
        "min_date": dates.min().strftime("%Y-%m-%d"),
        "max_date": dates.max().strftime("%Y-%m-%d"),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().items()},
        "quality_check_counts": {k: int(v.sum()) for k, v in checks.items()},
        "rows_sorted_by_date_within_every_location": all(dates[g.index].is_monotonic_increasing for _, g in df.groupby("province")),
        "minimums": {k: float(v) for k, v in numeric.min().items()},
        "maximums": {k: float(v) for k, v in numeric.max().items()},
    }
    cities = json.loads((ROOT / "config/cities.json").read_text(encoding="utf-8"))
    names = [c["kaggle_name"] for c in cities]
    selected = df[df.province.isin(names)].copy()
    if set(names) != set(selected.province.unique()):
        raise ValueError("A requested location is absent")
    selected = selected.sort_values(["province", "date"], kind="stable")
    write_csv(selected, data_dir / "weather_raw.csv")
    write_csv(df[df.province.isin(["Ha Noi", "Hanoi"])], data_dir / "hanoi_alias_raw.csv")
    coverage, missing_dates, years, city_stats = [], [], [], []
    for scope, frame in [("all_locations", df), ("project_three_cities", selected)]:
        for location, group in frame.groupby("province", sort=True):
            t = pd.to_datetime(group.date, format="%Y-%m-%d")
            expected = pd.date_range(t.min(), t.max(), freq="D")
            gaps = expected.difference(pd.DatetimeIndex(t))
            coverage.append({"scope": scope, "province": location, "rows": len(group),
                             "start_date": str(t.min().date()), "end_date": str(t.max().date()),
                             "expected_days": len(expected), "missing_days": len(gaps),
                             "coverage_pct": round(t.nunique() / len(expected) * 100, 5)})
            for day in gaps:
                missing_dates.append({"scope": scope, "province": location, "missing_date": str(day.date())})
            for year, n in t.dt.year.value_counts(sort=False).sort_index().items():
                years.append({"scope": scope, "province": location, "year": int(year), "rows": int(n)})
    a = df[df.province == "Ha Noi"].set_index("date").drop(columns="province").sort_index()
    b = df[df.province == "Hanoi"].set_index("date").drop(columns="province").sort_index()
    comparison = a.join(b, lsuffix="_Ha_Noi", rsuffix="_Hanoi", how="outer")
    conflicts = []
    for date, row in comparison.iterrows():
        for col in [c for c in EXPECTED_COLUMNS if c not in ["date", "province"]]:
            av, bv = row[col + "_Ha_Noi"], row[col + "_Hanoi"]
            if pd.isna(av) != pd.isna(bv) or (pd.notna(av) and av != bv):
                conflicts.append({"date": date, "column": col, "Ha_Noi": av, "Hanoi": bv})
    write_csv(pd.DataFrame(conflicts, columns=["date", "column", "Ha_Noi", "Hanoi"]), meta / "hanoi_conflicts.csv")
    for location, group in selected.groupby("province"):
        for col in NUMERIC:
            series = pd.to_numeric(group[col], errors="coerce")
            city_stats.append({"province": location, "column": col, "min": series.min(),
                               "max": series.max(), "mean": round(series.mean(), 4),
                               "median": series.median(), "missing_count": int(series.isna().sum())})
    write_csv(pd.DataFrame(coverage), meta / "coverage_by_city.csv")
    write_csv(pd.DataFrame(missing_dates), meta / "missing_dates.csv")
    write_csv(pd.DataFrame(years), meta / "rows_by_year.csv")
    write_csv(pd.DataFrame(city_stats), meta / "city_statistics.csv")
    write_csv(pd.DataFrame([{"check": k, "count": int(v.sum())} for k, v in checks.items()]), meta / "quality_checks.csv")
    write_csv(df.groupby("province", as_index=False).size().rename(columns={"size": "rows"}), meta / "all_location_counts.csv")
    write_csv(pd.DataFrame([{"compass": direction, "degrees": i * 22.5} for i, direction in enumerate(COMPASS)]), meta / "wind_direction_reference.csv")
    summary["project_rows"] = len(selected)
    summary["project_cities_original_names"] = names
    summary["hanoi_alias_compared_dates"] = len(comparison)
    summary["hanoi_alias_conflicting_cells"] = len(conflicts)
    summary["hanoi_alias_priority"] = "Select the entire Ha Noi series; retain Hanoi separately; no averaging or row-wise mixing."
    summary["unchanged_measurements"] = True
    dump_json(meta / "quality_summary.json", summary)
    # Xuất toàn bộ dòng nghi lỗi để người phụ trách tiền xử lý quyết định.
    bad_mask = pd.Series(False, index=df.index)
    for mask in checks.values():
        bad_mask |= mask
    write_csv(df[bad_mask], meta / "flagged_raw_rows.csv")
    print(json.dumps({"raw_rows": len(df), "project_rows": len(selected),
                      "quality_checks": summary["quality_check_counts"],
                      "hanoi_conflicting_cells": len(conflicts),
                      "source_sha256": summary["source_sha256"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
