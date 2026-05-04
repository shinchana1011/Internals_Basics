import os
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

EXPERIMENT_NAME = "pathscan-report-turnaround-hours"

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

df = pd.read_csv("data/training_data.csv")

X = df[["sample_count", "test_complexity", "lab_technician_count", "is_urgent"]]
y = df["report_turnaround_hours"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment(EXPERIMENT_NAME)

models = {
    "SVR": SVR(C=100, kernel="rbf", epsilon=0.1),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}

results = []
best_model_name = None
best_rmse = float("inf")
best_model = None
best_run_id = None

for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        params = model.get_params()
        mlflow.log_params(params)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.set_tag("priority", "high")
        mlflow.sklearn.log_model(model, artifact_path="model")

        results.append({
            "name": name,
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4)
        })

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_model = model
            best_run_id = run.info.run_id

joblib.dump(best_model, "models/champion_model.pkl")

metadata = {
    "best_model": best_model_name,
    "best_run_id": best_run_id,
    "best_rmse": float(best_rmse)
}

with open("models/champion_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

output = {
    "experiment_name": EXPERIMENT_NAME,
    "models": results,
    "best_model": best_model_name,
    "best_metric_name": "rmse",
    "best_metric_value": round(float(best_rmse), 4)
}

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)
print(json.dumps(output, indent=4))