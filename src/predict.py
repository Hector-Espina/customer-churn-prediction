"""Inferencia del modelo de predicción de churn.

Este módulo carga el modelo previamente entrenado y permite generar
predicciones para nuevos clientes.
"""

import pickle
from pathlib import Path

import pandas as pd

from src.preprocessing import clean_features


# Rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"


# Carga del modelo
def load_model():
    """Carga el modelo entrenado y el threshold desde el archivo pickle."""

    with open(MODEL_PATH, "rb") as file:
        artifact = pickle.load(file)

    return artifact["model"], artifact["threshold"]


# Cargamos el modelo una sola vez al importar el módulo.
model, threshold = load_model()


# Inferencia
def predict_churn(customer: dict) -> dict:

    customer_df = pd.DataFrame([customer])

    customer_df = clean_features(customer_df)

    churn_probability = model.predict_proba(customer_df)[0, 1]

    churn_prediction = int(churn_probability >= threshold)

    return {
        "churn_probability": float(churn_probability),
        "churn_prediction": churn_prediction,
        "threshold": float(threshold),
    }