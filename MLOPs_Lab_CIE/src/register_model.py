import os
import json
import mlflow
from mlflow.tracking import MlflowClient

REGISTERED_MODEL_NAME = "pathscan-report-turnaround-hours-predictor"

os.makedirs("results", exist_ok=True)

with open("models/champion_metadata.json", "r") as f:
    metadata = json.load(f)

run_id = metadata["best_run_id"]
rmse = metadata["best_rmse"]

model_uri = f"runs:/{run_id}/model"

result = mlflow.register_model(
    model_uri=model_uri,
    name=REGISTERED_MODEL_NAME
)

client = MlflowClient()

output = {
    "registered_model_name": REGISTERED_MODEL_NAME,
    "version": int(result.version),
    "run_id": run_id,
    "source_metric": "rmse",
    "source_metric_value": round(float(rmse), 4)
}

with open("results/step3_s6.json", "w") as f:
    json.dump(output, f, indent=4)

print(json.dumps(output, indent=4))