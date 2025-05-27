import os
import pickle
import click
import mlflow
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.ensemble import RandomForestRegressor
from hyperopt.pyll import scope
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("random-forest-hyperopt")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

# Load data
X_train, y_train = load_pickle(os.path.join("../output", "train.pkl"))
X_val, y_val = load_pickle(os.path.join("../output", "val.pkl"))



def objective(params):
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf.predict(X_val)
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        # Log RMSE
        mlflow.log_metric("rmse", rmse)
        
    return {'loss': rmse, 'status': STATUS_OK}



# Define search space
search_space = {
    'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
    'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
    'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
    'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
    'random_state': 42
}


# Run optimization
best_result = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=Trials()
)

print("Best result:", best_result)