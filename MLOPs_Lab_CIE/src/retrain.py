import os
import json
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

EXPERIMENT_NAME = "pathscan-report-turnaround-hours"
THRESHOLD = 0.3

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

train_df = pd.read_csv("data/training_data.csv")
new_df = pd.read_csv("data/new_data.csv")
combined_df = pd.concat([train_df, new_df], ignore_index=True)

with open("models/champion_metadata.json", "r") as f:
    metadata = json.load(f)

best_model_type = metadata["best_model"]

features = ["sample_count", "test_complexity", "lab_technician_count", "is_urgent"]
target = "report_turnaround_hours"

X_original = train_df[features]
y_original = train_df[target]

X_train_old, X_test, y_train_old, y_test = train_test_split(
    X_original, y_original, test_size=0.2, random_state=42
)

champion_model = joblib.load("models/champion_model.pkl")
champion_preds = champion_model.predict(X_test)
champion_mae = mean_absolute_error(y_test, champion_preds)

X_combined = combined_df[features]
y_combined = combined_df[target]

X_train_new, _, y_train_new, _ = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=42
)

if best_model_type == "SVR":
    retrained_model = SVR(C=100, kernel="rbf", epsilon=0.1)
else:
    retrained_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="Retraining-Pipeline"):
    retrained_model.fit(X_train_new, y_train_new)
    retrained_preds = retrained_model.predict(X_test)

    retrained_mae = mean_absolute_error(y_test, retrained_preds)
    improvement = champion_mae - retrained_mae

    mlflow.log_param("model_type", best_model_type)
    mlflow.log_metric("champion_mae", champion_mae)
    mlflow.log_metric("retrained_mae", retrained_mae)
    mlflow.log_metric("improvement", improvement)
    mlflow.set_tag("priority", "high")
    mlflow.sklearn.log_model(retrained_model, artifact_path="retrained_model")

if improvement >= THRESHOLD:
    action = "promoted"
    joblib.dump(retrained_model, "models/champion_model.pkl")
else:
    action = "kept_champion"

output = {
    "original_data_rows": int(len(train_df)),
    "new_data_rows": int(len(new_df)),
    "combined_data_rows": int(len(combined_df)),
    "champion_mae": round(float(champion_mae), 4),
    "retrained_mae": round(float(retrained_mae), 4),
    "improvement": round(float(improvement), 4),
    "min_improvement_threshold": THRESHOLD,
    "action": action,
    "comparison_metric": "mae"
}

with open("results/step4_s8.json", "w") as f:
    json.dump(output, f, indent=4)

print(json.dumps(output, indent=4))