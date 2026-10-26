# Thành viên 1 Thu thập và kiểm tra dữ liệu

Gói riêng cho thành viên 1, có dữ liệu Kaggle gốc. Dùng Python từ 3.11.

```bash
python -m pip install -r requirements.txt
python prepare_handover.py
```

Đầu ra chính: `data/interim/weather_raw.csv` và `reports/member1/`.
Bàn giao file weather_raw.csv cho thành viên 2. File gốc có 181.960 dòng; bảng ba thành phố có 13.647 dòng.

Để lấy mẫu OpenWeather hiện tại, sao chép `.env.example` thành `.env`, điền key trên máy rồi chạy:

```bash
python collect_openweather.py
```

`.env` được Git bỏ qua. Mẫu API hiện tại lưu riêng trong data/openweather_current, không ghép vào dữ liệu lịch sử theo ngày.

Nguồn: https://www.kaggle.com/datasets/vanviethieuanh/vietnam-weather-data
Metadata của nguồn tại 25/10/2026 ghi CC0 Public Domain. Giai đoạn: 01/01/2009–18/06/2021.
Gió km/h; áp suất có ghi chú đơn vị chưa nhất quán. Các ngày nhuận bị thiếu được ghi trong báo cáo kiểm tra.
Ha Noi được chọn nhất quán, Hanoi giữ riêng để đối chiếu.
