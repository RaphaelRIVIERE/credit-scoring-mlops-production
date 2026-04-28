import os
import joblib
import mlflow.pyfunc

model = None


def load(path: str):
    global model
    model = mlflow.pyfunc.load_model(path)


def load_joblib(path: str):
    """Charge model.pkl avec joblib — plus rapide que mlflow.pyfunc."""
    global model
    model = joblib.load(os.path.join(path, "model.pkl"))
