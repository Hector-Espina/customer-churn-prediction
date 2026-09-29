"""Preprocesamiento de datos para el modelo de predicción de churn.

Este módulo contiene las transformaciones necesarias para convertir los datos
en crudo al formato esperado por el modelo para realizar inferencia.
"""

import pandas as pd


# ---------------------------------------------------------------------
# Definición de columnas
# ---------------------------------------------------------------------

# Variables binarias con valores "Yes" y "No".
BINARY_COLUMNS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

# Servicios adicionales asociados a la conexión a Internet.
INTERNET_ADDON_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Variables categóricas nominales que posteriormente serán transformadas
# mediante One-Hot Encoding.
NOMINAL_COLUMNS = [
    "InternetService",
    "Contract",
    "PaymentMethod",
]


def clean_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    ).fillna(0)

    if "customerID" in df.columns:
        df = df.drop(columns="customerID")

    df["gender"] = df["gender"].map({
        "Female": 0,
        "Male": 1,
    })

    for column in BINARY_COLUMNS:
        df[column] = df[column].map({
            "No": 0,
            "Yes": 1,
        })

    df["MultipleLines"] = df["MultipleLines"].map({
        "No phone service": 0,
        "No": 0,
        "Yes": 1,
    })

    for column in INTERNET_ADDON_COLUMNS:
        df[column] = df[column].map({
            "No internet service": 0,
            "No": 0,
            "Yes": 1,
        })

    return df