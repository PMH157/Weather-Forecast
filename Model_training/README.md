# Thành viên 3 Huấn luyện và lựa chọn mô hình

Gói này cung cấp code để bạn tự train. Chưa có mô hình đã huấn luyện hoặc số đo Accuracy/F1 thực nghiệm trong gói.
Dùng Python từ 3.11. Có sẵn Train và Validation đã nhận từ thành viên 2.

Các script tự cấu hình stdout/stderr để ký tự terminal không biểu diễn được
hiện dưới dạng `\u...`, tránh UnicodeEncodeError trên Windows cp1252 và VS Code.
Terminal UTF-8 vẫn hiển thị tiếng Việt bình thường. Các file CSV/JSON luôn giữ
mã hóa UTF-8 như trước. Không cần đặt biến môi trường để xử lý lỗi in log này.

## Chạy

```bash
python -m pip install -r requirements.txt
python train.py --check-data
python train.py
```

`--check-data` chỉ kiểm tra đầu vào và cấu hình, không train.
`python train.py` mới bắt đầu huấn luyện. Cấu hình mặc định có 30 tổ hợp cho năm thuật toán.
Nếu muốn thử một cấu hình cho mỗi thuật toán trước:

```bash
python train.py --quick
```

Hoặc chỉ train các thuật toán bạn chọn:

```bash
python train.py --models logistic_regression random_forest
```

## Trực quan hóa quá trình huấn luyện

`python train.py` tự tạo biểu đồ sau khi chọn và lưu mô hình cuối. Phần này train thêm
các mô hình chẩn đoán bằng cấu hình tốt nhất của từng thuật toán, nên mất thêm thời gian.
Code không tự train khi import; gói tải xuống chưa chứa kết quả train hoặc biểu đồ thực nghiệm.

Mỗi hình có bốn ô: Accuracy, Macro F1, Loss và thời gian fit. Ba ô số đo có hai đường
Train và Validation. Mặc định có tối đa bảy hình, tùy các thuật toán train thành công:

| Biểu đồ | Trục X | Loss và ý nghĩa |
|---|---|---|
| Learning curve cho cả năm thuật toán | Số mẫu Train tăng theo thời gian: khoảng 10%, 25%, 50%, 75%, 100% | Log loss cho mô hình có predict_proba; riêng SVM dùng classification error = 1 − Accuracy |
| Logistic Regression: iteration checkpoints | Số vòng lặp L-BFGS thực tế tại mỗi mốc | Predictive log loss trên Train và Validation |
| Random Forest: tree growth | Số cây tăng dần đến số cây trong cấu hình được chọn | Predictive log loss trên Train và Validation |

KNN và Decision Tree không có epoch như mạng neural. Learning curve của chúng dùng
số mẫu, không gán nhãn trục là Epoch. SVC hiện đặt `probability=False`, vì vậy không
biến điểm decision_function thành xác suất để tính log loss.

Đường Logistic Regression được tái hiện bằng các lần fit độc lập từ cùng khởi tạo,
cùng Train, solver và cấu hình; mỗi lần tăng `max_iter`. Trục X lấy `n_iter_` thực tế
và dừng khi hội tụ. Đây không phải log trực tiếp sau từng epoch của một lần fit duy nhất.
Log loss được vẽ là cross entropy không trọng số của dự đoán, không gồm regularization
hoặc class weights trong mục tiêu tối ưu của solver.

Random Forest dùng `warm_start=True` để thêm cây trên cùng Train với cùng bộ tiền xử lý.
Thời gian ở ô cuối là thời gian fit tích lũy, không gồm thời gian dự đoán/vẽ. Ở learning
curve và Logistic Regression, ô thời gian là thời lượng fit riêng tại từng mốc.

Learning curve giữ nguyên Validation và lấy các prefix ngày của Train, giữ toàn bộ
thành phố cùng ngày trong một mốc. Mỗi prefix tự học trung vị và scaler từ chính prefix đó.
Mốc thiếu một trong ba lớp hoặc không đủ láng giềng KNN được bỏ qua. Vì các prefix cũng
khác giai đoạn thời tiết, thay đổi số đo có thể phản ánh cả mùa vụ/phân bố theo thời gian.

File tạo tại `runs/TEN_LAN_CHAY/results/visualizations/`:

```text
logistic_regression_learning_curve.png / .csv
logistic_regression_iteration_checkpoints.png / .csv
knn_learning_curve.png / .csv
decision_tree_learning_curve.png / .csv
random_forest_learning_curve.png / .csv
random_forest_tree_growth.png / .csv
svm_learning_curve.png / .csv
curves_index.csv
visualization_manifest.json
```

Mỗi CSV lưu số đo từng mốc và cảnh báo hội tụ nếu có. `curves_index.csv` ghi rõ hình nào
thành công hoặc gặp lỗi. Lỗi vẽ không làm mất Pipeline cuối đã lưu.

Để train trước và bỏ qua phần biểu đồ:

```bash
python train.py --no-plots
```

Để tạo lại biểu đồ từ cấu hình đã chọn của một lần chạy, không tìm kiếm lại 30 tổ hợp:

```bash
python training_visualization.py --run-dir runs/TEN_LAN_CHAY --settings config/training.json
```

Lệnh trên vẫn train lại các mô hình chẩn đoán để lấy số đo tại từng mốc. Nó kiểm tra
hash đầu vào khớp lần train đã chọn, không đọc Test và không sửa Pipeline cuối.
Chỉnh các mốc hoặc bật/tắt từng loại hình ở khóa `visualization` trong `config/training.json`.
`--quick` giảm số cấu hình tìm kiếm; muốn thử nhanh hơn nữa có thể giảm số mốc vẽ.

### Đọc biểu đồ

- Dấu hiệu overfitting: Train có Accuracy/F1 cao nhưng Validation thấp hơn nhiều; hoặc Train loss tiếp tục giảm trong khi Validation loss tăng.
- Dấu hiệu underfitting: cả Train và Validation đều có số đo thấp so với baseline phù hợp và loss còn cao, kể cả khi tăng dữ liệu hoặc số vòng lặp.
- Hai đường gần nhau chưa đủ kết luận mô hình tốt; cần đối chiếu baseline, F1 từng lớp và ma trận nhầm lẫn. Không dùng một ngưỡng chênh lệch cố định để tự kết luận.

## Các thuật toán

| Thuật toán | Nội dung thử nghiệm |
|---|---|
| Logistic Regression | C và trọng số lớp |
| KNN | Số láng giềng, trọng số uniform/distance |
| Decision Tree | Độ sâu, số mẫu tại lá, trọng số lớp |
| Random Forest | Số cây, độ sâu, số mẫu tại lá, trọng số lớp |
| SVM bằng SVC | C, kernel linear/RBF, trọng số lớp |

Có hai baseline trên Validation: luôn chọn lớp phổ biến nhất trong Train và dự báo ngày mai giống hôm nay.
Không dùng baseline để tự thay đổi nhãn hoặc tập dữ liệu.

## Dữ liệu bàn giao từ phần 2

| File trong data | Vai trò |
|---|---|
| train.csv | 8.181 mẫu trước imputation |
| validation.csv | 2.043 mẫu trước imputation |
| feature_schema.json | Thứ tự cột X và ánh xạ nhãn |
| preprocessing_config.json | Cấu hình tạo nhãn của phần 2 để đối chiếu |

Bạn có thể thay các file này bằng đầu ra mới của thành viên 2, giữ đúng tên và cấu trúc.
Không dùng weather_prepared.csv đã chuẩn hóa sẵn làm đầu vào cho script này.
Các ô NaN ở đặc trưng lịch sử là hợp lệ; Pipeline sẽ học trung vị từ Train.
Chỉ lấy đúng feature_columns từ schema; nhãn, ngày định danh và weather_today_code không được đưa vào X.

## Quy tắc train và chọn mô hình

1. Mỗi cấu hình tạo một Pipeline mới: điền thiếu, mã hóa và classifier. Logistic Regression, KNN, SVM có thêm StandardScaler.
2. Pipeline chỉ fit trên Train, sau đó predict Validation.
3. So sánh bằng Macro F1 trên Validation. Nếu bằng nhau, ưu tiên Accuracy rồi mã cấu hình để kết quả lựa chọn xác định.
4. Lưu dự đoán, báo cáo từng lớp và ma trận nhầm lẫn trên Validation của mô hình được chọn khi chỉ fit Train.
5. Clone Pipeline đã chọn và fit lại trên Train + Validation để bàn giao.

Không đọc Test trong quá trình train hoặc lựa chọn. Test do thành viên 4 giữ từ bộ của thành viên 2.
Các nhãn 0 Nắng hoặc ít mây, 1 Nhiều mây, 2 Mưa là quy ước của đồ án trên dữ liệu Kaggle, không phải nhãn quan trắc đã xác minh độc lập.

## Kết quả sau khi bạn chạy train

Mỗi lần chạy mặc định tạo một thư mục riêng `runs/<thời_điểm>/`:

| File | Nội dung |
|---|---|
| results/all_candidates_validation.csv | Tiến trình, số đo và tham số của từng cấu hình |
| results/validation_ranking.csv | Xếp hạng các cấu hình thành công |
| results/best_by_model.csv | Cấu hình tốt nhất của từng thuật toán |
| results/baselines_validation.csv | Hai mốc đối chiếu |
| results/winner_validation_predictions.csv | Dự đoán Validation trước khi refit Train+Validation |
| results/winner_validation_confusion_matrix.csv | Ma trận nhầm lẫn Validation |
| results/winner_validation_classification_report.json | Precision, Recall, F1 của từng lớp |
| results/search_errors.json | Các cấu hình gặp lỗi nếu có |
| models/final_pipeline.joblib | Toàn bộ Pipeline cuối đã fit trên Train+Validation |
| models/feature_schema.json | Danh sách đầu vào và ánh xạ nhãn |
| models/selection.json | Tham số đã chọn, kết quả Validation, nguồn dữ liệu và phiên bản thư viện |

Bàn giao cả thư mục models cùng kết quả Validation cho thành viên 4. Số đo Validation trong selection.json thuộc mô hình fit riêng trên Train, không phải kết quả đánh giá Pipeline sau refit trên chính Validation.
Nếu muốn chọn nơi lưu, thêm `--output-dir duong_dan_moi`. Thư mục kết quả cần trống để không lẫn các lần chạy.

## Dùng mô hình sau khi train

```bash
python predict.py --model runs/TEN_LAN_CHAY/models/final_pipeline.joblib --input your_feature_rows.csv --output predictions.csv
```

CSV đầu vào phải có đầy đủ các feature_columns đã được tạo giống phần 2, gồm các biến trễ và rolling.
Không đưa trực tiếp một phản hồi OpenWeather hiện tại vào Pipeline theo ngày này.
Script dự đoán xuất nhãn, không tự đánh giá Test và không diễn giải điểm SVM thành xác suất.

## Thay cấu hình

Chỉnh `config/training.json`. Các tên tham số có tiền tố classifier__, ví dụ classifier__C hoặc classifier__max_depth.
Không điều chỉnh cấu hình dựa vào Test. Nếu Train còn mất cân bằng lớp, so sánh trọng số lớp bằng Validation.
Nếu kết quả chưa vượt baseline, cần báo cáo đúng kết quả đó, không chỉ trình bày Accuracy.

Nguồn dữ liệu: https://www.kaggle.com/datasets/vanviethieuanh/vietnam-weather-data
Tài liệu Pipeline: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
Tài liệu F1: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
Tài liệu Logistic Regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
Tài liệu Random Forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
Tài liệu log loss: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
