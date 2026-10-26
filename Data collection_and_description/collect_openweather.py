"""Lưu dữ liệu hiện tại của OpenWeather thành JSON gốc và CSV riêng.

Không đưa các bản ghi tại một thời điểm này vào bảng lịch sử theo ngày.
Chỉ gọi Current Weather API, không sử dụng endpoint lịch sử trả phí.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import socket
import sys
from datetime import datetime, timezone, timedelta
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent
LOCAL_TZ = timezone(timedelta(hours=7))
ENDPOINT = "https://api.openweathermap.org/data/2.5/weather"


def flatten(payload, city, retrieved_at):
    """Giữ các trường thiếu là None, không điền giả lượng mưa bằng 0."""
    main = payload.get("main", {})
    wind = payload.get("wind", {})
    weather = (payload.get("weather") or [{}])[0]
    timestamp = payload.get("dt")
    instant = datetime.fromtimestamp(timestamp, timezone.utc) if timestamp is not None else None
    return {
        "requested_city": city["city"],
        "provider_location_name": payload.get("name"),
        "requested_lat": city["lat"], "requested_lon": city["lon"],
        "returned_lat": payload.get("coord", {}).get("lat"),
        "returned_lon": payload.get("coord", {}).get("lon"),
        "observed_at_utc": instant.isoformat() if instant else None,
        "observed_at_local": instant.astimezone(LOCAL_TZ).isoformat() if instant else None,
        "retrieved_at_utc": retrieved_at,
        "temperature_c": main.get("temp"),
        "temp_min_at_moment_c": main.get("temp_min"),
        "temp_max_at_moment_c": main.get("temp_max"),
        "humidity_pct": main.get("humidity"),
        "pressure_hpa": main.get("pressure"),
        "wind_speed_m_s": wind.get("speed"), "wind_direction_deg": wind.get("deg"),
        "cloud_cover_pct": payload.get("clouds", {}).get("all"),
        "rain_1h_mm_per_h": payload.get("rain", {}).get("1h"),
        "rain_1h_field_present": "1h" in payload.get("rain", {}),
        "weather_code": weather.get("id"), "weather_main": weather.get("main"),
        "weather_description": weather.get("description"),
        "source": "OpenWeather Current Weather API", "units": "metric",
        "time_resolution": "instant_snapshot",
    }


def collect(out_dir, api_key, timeout=20):
    cities = json.loads((ROOT / "config/cities.json").read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows, statuses = [], []
    for city in cities:
        params = {"lat": city["lat"], "lon": city["lon"], "appid": api_key,
                  "units": "metric", "lang": "vi"}
        request = urllib.request.Request(ENDPOINT + "?" + urllib.parse.urlencode(params),
                                         headers={"User-Agent": "WeatherCourseProject/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                statuses.append({"city": city["city"], "success": False,
                                 "error": "API JSON must be an object"})
                continue
            if str(payload.get("cod")) != "200":
                statuses.append({"city": city["city"], "success": False,
                                 "error": "API response not successful", "code": payload.get("cod")})
                continue
            if not isinstance(payload.get("main"), dict) or payload.get("dt") is None:
                statuses.append({"city": city["city"], "success": False,
                                 "error": "Missing main or dt in API response"})
                continue
            retrieved_at = datetime.now(timezone.utc).isoformat()
            safe_name = city["kaggle_name"].lower().replace(" ", "_")
            json_path = raw_dir / f"{safe_name}_{run_id}.json"
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append(flatten(payload, city, retrieved_at))
            statuses.append({"city": city["city"], "success": True, "http_status": 200,
                             "raw_file": str(json_path.relative_to(out_dir))})
        except urllib.error.HTTPError as error:
            explanation = {401: "API key not accepted for this endpoint", 403: "Access denied",
                           429: "Rate limit reached"}.get(error.code, "HTTP request failed")
            statuses.append({"city": city["city"], "success": False,
                             "http_status": error.code, "error": explanation})
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError):
            statuses.append({"city": city["city"], "success": False,
                             "error": "Connection or timeout error"})
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError, OverflowError):
            statuses.append({"city": city["city"], "success": False,
                             "error": "Invalid API JSON or field types"})
        # Không in URL request hoặc nội dung exception vì có thể chứa API key.
    csv_path = None
    if rows:
        csv_path = out_dir / f"openweather_current_{run_id}.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {"run_id_utc": run_id, "endpoint": ENDPOINT,
               "requested_count": len(cities), "success_count": len(rows),
               "csv_file": csv_path.name if csv_path else None, "cities": statuses,
               "note": "Current snapshots are separate from Kaggle daily history. No daily aggregation performed."}
    (out_dir / f"collection_log_{run_id}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return len(rows) == len(cities)


def load_api_key(env_path=None):
    """Ưu tiên biến môi trường, sau đó đọc đúng biến này trong .env."""
    key = os.environ.get("OPENWEATHER_API_KEY", "").strip()
    if key and key != "YOUR_KEY_HERE":
        return key
    env_path = Path(env_path) if env_path is not None else ROOT / ".env"
    if not env_path.is_file():
        return ""
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        name, separator, value = line.partition("=")
        if separator and name.strip() == "OPENWEATHER_API_KEY":
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", chr(34)):
                value = value[1:-1]
            return "" if value == "YOUR_KEY_HERE" else value
    return ""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/openweather_current")
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args(argv)
    try:
        key = load_api_key()
    except OSError:
        print("Cannot read .env file.", file=sys.stderr)
        return 2
    if not key:
        print("Set OPENWEATHER_API_KEY in your environment or .env before running.", file=sys.stderr)
        return 2
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return 0 if collect(args.output, key, args.timeout) else 1


if __name__ == "__main__":
    sys.exit(main())
