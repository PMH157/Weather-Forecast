# Dự báo thời tiết bằng Random Forest

Dự báo cho **Hà Nội, TP. Hồ Chí Minh, Nha Trang** trong **3 ngày tới**: loại thời tiết (Nắng hoặc ít mây /
Nhiều mây / Mưa), nhiệt độ cao nhất, nhiệt độ thấp nhất và lượng mưa.

Có hai cách dùng:
- **Thời gian thực** (trang web mặc định, hoặc `python predict.py --live`): lấy số liệu mới nhất từ Open-Meteo
  đến hết hôm qua, dự báo **hôm nay, ngày mai, ngày kia**. Cần Internet.
- **Từ file CSV:** dự báo 3 ngày sau ngày cuối cùng trong file, ví dụ `data/weather.csv` (Kaggle, 2009–2021).

```
Ho Chi Minh City - dự báo từ ngày 2021-06-18
  19/06: Mưa              26-33°C  mưa ~6.9 mm  (độ tin cậy cao)
  20/06: Mưa              26-33°C  mưa ~9.1 mm  (độ tin cậy cao)
  21/06: Mưa              26-32°C  mưa ~10.5 mm  (độ tin cậy cao)
```

## Cài đặt

Cần Python 3.11 trở lên. Trong thư mục `forecast_3days`:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
source .venv/bin/activate         # macOS / Linux
pip install -r requirements.txt
```

## Chạy

| Lệnh | Việc làm | Thời gian (máy 4 nhân) |
|---|---|---|
| `python -m unittest discover -s tests -v` | 52 test tự động (không cần Internet) | ~20 giây |
| `python explore.py` | Mô tả dữ liệu, biểu đồ tỷ lệ thời tiết theo tháng → `outputs/explore/` | ~5 giây |
| `python train.py --quick` | Thử code chạy được (1 tổ hợp, 50 cây, không Golden Section, **không chấm Test**) | ~10 giây |
| `python train.py` | Tìm tham số → fit lại trên 80% → chấm Test 20% một lần → `runs/<ngày_giờ>/` | ~27 phút |
| `python open_meteo.py` | Tải số liệu Open-Meteo 2009 → hôm qua vào `data/open_meteo.csv` (đã có sẵn đến 08/11/2026) | ~10 giây |
| `python train.py --data data/open_meteo.csv --targets tmax tmin rain` | Train 9 mô hình nhiệt độ và lượng mưa trên Open-Meteo, dùng cho chế độ thời gian thực | ~70 phút |
| `python evaluate_live.py` | Đo accuracy của mô hình loại thời tiết khi nhận số liệu Open-Meteo | ~10 giây |
| `python predict.py --live` | In bản tin hôm nay, ngày mai, ngày kia (cần Internet) | ~10 giây |
| `python predict.py` | In bản tin 3 ngày sau ngày cuối của `data/weather.csv`, ghi `forecast.csv` | ~5 giây |
| `streamlit run app.py` | Mở **trang web dự báo** trong trình duyệt | mở ngay |

Tuỳ chọn:
- `--data đường_dẫn.csv` dùng file dữ liệu khác, áp dụng cho cả 3 script.
- `--out thư_mục` chọn nơi lưu kết quả của `train.py`. Thư mục phải trống.
- `python train.py --final-only runs/<ngày_giờ>` chỉ fit lại và chấm Test cho một lần chạy đã tìm tham số. Lệnh này
  từ chối chạy nếu Test của thư mục đó đã được chấm, để mỗi lần chạy chỉ chấm Test một lần. Kiểm tra này chỉ áp dụng
  cho từng thư mục: chạy lại với `--out` mới vẫn chấm Test được. Vì vậy đừng chỉnh mô hình rồi chạy lại để nhìn điểm Test.

## Trang web dự báo

```bash
streamlit run app.py
```

Trình duyệt tự mở `http://localhost:8501`. Nếu không tự mở, dán địa chỉ này vào trình duyệt.

Thứ tự chuẩn bị, chỉ cần làm một lần:
1. `python train.py`: mô hình loại thời tiết (học từ Kaggle).
2. `python train.py --data data/open_meteo.csv --targets tmax tmin rain`: mô hình nhiệt độ và lượng mưa (học từ Open-Meteo).
3. `python evaluate_live.py`: đo độ chính xác để trang web hiển thị.

Trang web gồm:
- **Chế độ thời gian thực (mặc định):** thẻ *Hôm nay / Ngày mai / Ngày kia* cho từng thành phố, gồm loại thời tiết,
  nhiệt độ thấp–cao, lượng mưa và độ tin cậy. Số liệu tự cập nhật sau mỗi giờ; nút "Cập nhật số liệu" lấy lại ngay.
- **Chế độ từ file CSV:** tải lên CSV cùng định dạng `data/weather.csv`, dự báo từ ngày cuối cùng trong file.
- Mục "Dự báo này chính xác đến đâu?": độ chính xác đã đo của từng phần.
- Mục "Bảng chi tiết": bảng dự báo, có nút tải về CSV.

Trang web chạy trên máy bạn. Tắt bằng `Ctrl + C` trong terminal. Chưa làm bước 2 thì nhiệt độ và lượng mưa dùng
tạm mô hình Kaggle, và trang có ghi chú. Mất Internet thì trang báo lỗi rõ ràng.

## Các file

| File | Vai trò |
|---|---|
| `features.py` | Đọc CSV, tạo nhãn ngày t+1..t+3, tạo 46 đặc trưng (chỉ dùng số liệu đến hết ngày t), chia tập theo thời gian |
| `explore.py` | Mô tả dữ liệu |
| `train.py` | Baseline, Grid Search, Golden Section, đánh giá, fit lại, chấm Test, lưu mô hình và biểu đồ |
| `open_meteo.py` | Lấy số liệu Open-Meteo (lịch sử và thời gian thực), đổi sang cùng định dạng `data/weather.csv`, tự thử lại khi bị giới hạn lượt gọi |
| `predict.py` | Bản tin dự báo; ghép loại thời tiết từ một lần chạy với nhiệt độ, lượng mưa từ lần chạy khác |
| `evaluate_live.py` | Đo mô hình loại thời tiết khi nhận số liệu Open-Meteo |
| `app.py` | Trang web dự báo (Streamlit) |
| `data/open_meteo.csv` | Số liệu Open-Meteo 2009-01-01 → 2026-11-08 cho 3 thành phố |
| `tests/` | Test tự động: chống rò rỉ dữ liệu tương lai, luật nhãn, chia tập, baseline, Golden Section, bản tin |

## Phương pháp

**Dữ liệu và nhãn.** Mỗi thành phố có 4.549 ngày, thiếu 3 ngày. Loại thời tiết của một ngày được gán như sau:
Mưa nếu `rain ≥ 1 mm`; Nhiều mây nếu không mưa và `cloud ≥ 50%`; còn lại là Nắng hoặc ít mây.

**Đặc trưng tại ngày t** (chỉ dùng số liệu ≤ t):
- giá trị hôm nay, gồm cả áp suất và hướng gió (sin/cos);
- giá trị 1–2 ngày trước và mức thay đổi so với hôm qua;
- trung bình 3 và 7 ngày, số ngày mưa trong 7 ngày;
- mã thời tiết hôm nay và hôm qua;
- mùa (sin/cos của ngày cần dự báo) và thành phố.

**Chia tập theo thời gian**, không chia ngẫu nhiên, vì thời tiết hôm nay rất giống hôm qua:

| Tập | Tỷ lệ | Ngày gốc | Dùng để |
|---|---|---|---|
| Train | 65% | 01/2009 → 02/2017 | Học |
| Validation | 15% | 02/2017 → 12/2018 | Chọn tham số |
| Test | 20% | 12/2018 → 06/2021 | Chấm điểm cuối, một lần |

Mô hình cuối học trên Train + Validation (80%). Ở mỗi ranh giới bỏ 3 ngày gốc, để ngày cần dự báo không lọt sang tập sau.

**12 mô hình.** Mỗi (đại lượng, ngày dự báo) có một Pipeline riêng: điền trung vị → one-hot thành phố → Random Forest.

**Baseline** để so sánh:
- "Giống hôm nay": ngày t+h giống ngày t.
- "Theo tháng": nhãn hay gặp nhất, hoặc giá trị trung bình, của (thành phố, tháng). Chỉ học từ dữ liệu huấn luyện.

## Kết quả

### Test (20%, chấm một lần), gộp 3 thành phố

| Đại lượng | Ngày +1 | Ngày +2 | Ngày +3 | Baseline mạnh nhất (+1 / +2 / +3) |
|---|---|---|---|---|
| Loại thời tiết (accuracy) | **76,0%** | **70,6%** | 68,7% | 72,8% / 68,8% / **68,8%** |
| Nhiệt độ cao (MAE) | **1,31°C** | **1,58°C** | **1,68°C** | 1,41 / 1,83 / 1,84°C |
| Nhiệt độ thấp (MAE) | **0,89°C** | **1,13°C** | **1,22°C** | 0,91 / 1,20 / 1,37°C |
| Lượng mưa (MAE) | **4,6 mm** | **5,2 mm** | **5,3 mm** | 5,0 / 5,5 / 5,5 mm |

- Random Forest vượt baseline mạnh nhất ở **11/12 ô**.
- **Loại thời tiết ngày +3 thua baseline "theo tháng" 0,1 điểm** (68,7% so với 68,8%). Từ ngày thứ 3, dữ liệu theo ngày
  gần như không còn đủ thông tin để đoán tốt hơn việc dựa vào mùa.
- Kết quả trên Validation là 77,1% / 71,5% / 69,0%, chênh với Test không quá 1,1 điểm, nên mô hình không bị overfit
  vào Validation.

**Chỉ số phụ cho loại thời tiết (Test):**

| | Ngày +1 | Ngày +2 | Ngày +3 |
|---|---|---|---|
| Mưa / Không mưa | 80,1% | 75,6% | 73,6% |
| Ngày "độ tin cậy cao" | 85,9% (trên 66,5% số ngày) | 79,8% (trên 67,0% số ngày) | 78,9% (trên 62,6% số ngày) |

Ngưỡng độ tin cậy được chọn trên Validation để giữ 70% số ngày. Trên Test, tỷ lệ số ngày đạt ngưỡng chỉ còn 63–67%.

**Theo thành phố (loại thời tiết, Test):**

| | Ngày +1 | Ngày +2 | Ngày +3 |
|---|---|---|---|
| TP. Hồ Chí Minh | 83,7% | 79,4% | 76,3% |
| Nha Trang | 77,9% | 74,1% | 71,9% |
| Hà Nội | 66,3% | 58,2% | 57,8% |

Hà Nội khó đoán nhất vì có khoảng 22% số ngày "Nhiều mây", chủ yếu từ tháng 12 đến tháng 4. TP. Hồ Chí Minh và
Nha Trang gần như chỉ có Nắng và Mưa, và theo mùa rõ rệt (xem `outputs/explore/class_share_by_city_month.png`).

**Ma trận nhầm lẫn ngày +1 (Test):** lớp "Nhiều mây" chỉ đoán đúng 137/340 ngày. Các ngày bị đoán sai chủ yếu bị nhầm
thành Mưa (126 ngày).

**Đặc trưng quan trọng nhất (ngày +1):** lượng mưa hôm nay, mã thời tiết hôm nay, lượng mây, lượng mưa trung bình
3 và 7 ngày, áp suất trung bình 7 ngày.

### Chế độ thời gian thực

| Phần | Mô hình | Ngày +1 | Ngày +2 | Ngày +3 |
|---|---|---|---|---|
| Loại thời tiết (accuracy) | Học từ Kaggle, đầu vào Open-Meteo | 69,7% | 70,3% | 69,5% |
| Nhiệt độ cao (sai số) | Học từ Open-Meteo | 1,04°C | 1,37°C | 1,53°C |
| Nhiệt độ thấp (sai số) | Học từ Open-Meteo | 0,59°C | 0,81°C | 0,95°C |
| Lượng mưa (sai số) | Học từ Open-Meteo | 4,5 mm | 5,2 mm | 5,4 mm |

- **Loại thời tiết** đo bằng `evaluate_live.py`: mô hình Kaggle nhận số liệu Open-Meteo trên giai đoạn Test của Kaggle
  (12/2018–06/2021), so với nhãn Kaggle. Đổi nguồn số liệu làm giảm khoảng 6 điểm ở ngày +1 (76,0% còn 69,7%), vì hai
  nguồn đo mây và mưa khác nhau. Ngày +2 và +3 gần như không đổi.
- **Các con số** đo trên Test của Open-Meteo (04/2023–11/2026, chấm một lần). Cả 9 mô hình đều vượt baseline mạnh nhất
  (nhiệt độ cao: baseline 1,10 / 1,50 / 1,74°C).
- Đã thử hiệu chỉnh số liệu Open-Meteo về thang của Kaggle (tuyến tính, theo phân vị): không làm loại thời tiết chính
  xác hơn, nên không dùng.
- Train lại cả loại thời tiết trên Open-Meteo cho accuracy thấp hơn (65,3% ngày +1), vì theo Open-Meteo có tới 30% số ngày
  "Nhiều mây", khó đoán hơn nhiều. Vì vậy loại thời tiết vẫn dùng mô hình Kaggle.

## Kết quả của một lần chạy (`runs/<ngày_giờ>/`)

| File | Nội dung |
|---|---|
| `results/validation_search.csv` | Mọi tổ hợp tham số đã thử và điểm Validation |
| `results/golden_vs_grid.csv` | Grid Search so với Golden Section |
| `results/leaf_curve.csv`, `figures/leaf_curve_h*.png` | Accuracy theo `min_samples_leaf` |
| `results/validation_vs_baseline.csv` | Validation: RF so với baseline, gộp và từng thành phố |
| `results/validation_extra.csv`, `results/test_extra.csv` | Mưa/Không mưa, độ tin cậy |
| `results/test_metrics.csv` | Test: RF so với baseline, gộp và từng thành phố |
| `results/confusion_h*.csv` | Ma trận nhầm lẫn trên Test |
| `results/feature_importance_*_h*.csv`, `figures/feature_importance_weather_h*.png` | Độ quan trọng đặc trưng |
| `figures/accuracy_by_horizon.png`, `figures/mae_by_horizon.png` | Độ chính xác giảm theo số ngày dự báo |
| `models/*.joblib`, `models/metadata.json` | Các mô hình đã fit (`runs/full` khoảng 160 MB, `runs/open_meteo` khoảng 240 MB) và thông tin để tái lập |

`runs/*/models/` nặng (khoảng 160 MB) và tạo lại được bằng `python train.py`, nên đã có trong `.gitignore`. Các file
`results/` và `figures/` vẫn được đưa lên GitHub làm minh chứng cho kết quả.

## Giới hạn

- Nhãn và các con số được dự báo bằng các mô hình riêng, nên có thể không khớp nhau. Ví dụ "Nắng" nhưng mưa ~2 mm:
  bản tin có ghi chú khi gặp trường hợp này.
- Mô hình lượng mưa dự báo giá trị trung bình, nên hiếm khi ra 0 mm và không bắt được những ngày mưa rất to.
- "Độ tin cậy" chỉ là xác suất lớn nhất của Random Forest, không phải xác suất đã được hiệu chỉnh.
- Nhãn được suy từ số liệu tổng hợp theo ngày của bộ dữ liệu Kaggle, không phải quan trắc khí tượng chính thức.
- `predict.py` dự báo từ ngày cuối của CSV. Thành phố thiếu số liệu ngày đó sẽ được báo và bỏ qua.

Nguồn dữ liệu: https://www.kaggle.com/datasets/vanviethieuanh/vietnam-weather-data
