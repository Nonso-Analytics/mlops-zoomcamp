import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

# Load test data
X_test, y_test = load_pickle("../output/test.pkl")

# Initialize MLflow client
client = MlflowClient("http://localhost:5000")
mlflow.set_tracking_uri("http://localhost:5000")

# Get the best runs from hyperopt experiment
experiment = client.get_experiment_by_name("random-forest-hyperopt")
best_runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.rmse ASC"],
    max_results=5
)

# Set up new experiment for best models
mlflow.set_experiment("random-forest-best-models")

best_rmse = float('inf')
best_run_id = None

for run in best_runs:
    with mlflow.start_run():
        # Load the model
        model = mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/model")
        
        # Predict on test set
        y_pred = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Log test RMSE
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_param("original_run_id", run.info.run_id)
        
        print(f"Run {run.info.run_id}: Test RMSE = {test_rmse}")
        
        # Track best model
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_run_id = run.info.run_id

print(f"\nBest model - Run ID: {best_run_id}, Test RMSE: {best_rmse}")

# Register the best model
model_uri = f"runs:/{best_run_id}/model"
model_name = "random-forest-best-model"

mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)

print(f"Model registered with name: {model_name}")
print(f"Best test RMSE: {best_rmse}")