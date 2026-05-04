import argparse
import json
import joblib
import pandas as pd
import os

MODEL_PATH = "models/champion_model.pkl"

parser = argparse.ArgumentParser()

parser.add_argument("--sample_count", type=float, required=True)
parser.add_argument("--test_complexity", type=float, required=True)
parser.add_argument("--lab_technician_count", type=float, required=True)
parser.add_argument("--is_urgent", type=float, required=True)

args = parser.parse_args()

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found. Run src/train.py first.")

model = joblib.load(MODEL_PATH)

input_data = pd.DataFrame([{
    "sample_count": args.sample_count,
    "test_complexity": args.test_complexity,
    "lab_technician_count": args.lab_technician_count,
    "is_urgent": args.is_urgent
}])

prediction = float(model.predict(input_data)[0])

output = {
    "prediction": round(prediction, 4)
}

print(json.dumps(output))

os.makedirs("results", exist_ok=True)

output_json = {
    "image_name": "pathscan-predictor",
    "image_tag": "v1",
    "base_image": "python:3.10-slim",
    "test_input": {
        "sample_count": args.sample_count,
        "test_complexity": args.test_complexity,
        "lab_technician_count": args.lab_technician_count,
        "is_urgent": args.is_urgent
    },
    "prediction": round(prediction, 4)
}

with open("results/step2_s3.json", "w") as f:
    json.dump(output_json, f, indent=4)