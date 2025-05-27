import os
import pickle
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

# Set MLflow tracking URI (optional for local)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("random-forest-train")

# Enable autologging but disable dataset logging to avoid memory issues
mlflow.sklearn.autolog(log_datasets=False)

with mlflow.start_run():
    # Load data
    X_train, y_train = load_pickle(os.path.join("../output", "train.pkl"))
    X_val, y_val = load_pickle(os.path.join("../output", "val.pkl"))
    
    # Train model
    rf = RandomForestRegressor(max_depth=10, random_state=0)
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_val)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"RMSE: {rmse}")