"""Đọc Pipeline người dùng đã train và dự đoán các dòng đặc trưng tương thích."""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from modeling import configure_console, read_json


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True, help='CSV có các feature_columns giống phần 2')
    parser.add_argument('--output', type=Path, default=Path('predictions.csv'))
    args = parser.parse_args()
    schema = read_json(args.model.parent/'feature_schema.json')
    data = pd.read_csv(args.input)
    missing = set(schema['feature_columns']) - set(data.columns)
    if missing:
        parser.error('Thiếu các đặc trưng: ' + ', '.join(sorted(missing)))
    for column in schema['numeric_features']:
        data[column] = pd.to_numeric(data[column], errors='raise')
    pipeline = joblib.load(args.model)
    prediction = pipeline.predict(data[schema['feature_columns']])
    result = data[[col for col in ['sample_id','city','origin_date','target_date'] if col in data]].copy()
    result['prediction'] = prediction
    code_to_label = {value:key for key,value in schema['label_to_code'].items()}
    result['label'] = [schema.get('label_display_vi',{}).get(code_to_label[int(code)],code_to_label[int(code)])
                       for code in prediction]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output,index=False,encoding='utf-8-sig')
    print(args.output)


if __name__ == '__main__':
    main()
