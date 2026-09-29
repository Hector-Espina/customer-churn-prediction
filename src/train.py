"""Entrenamiento del modelo final de predicción de churn.

Este módulo carga los datos originales, aplica el preprocesamiento definido
para el proyecto, entrena el modelo final y guarda los artefactos necesarios
para realizar inferencia posteriormente.
"""

import pickle
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier

from src.preprocessing import clean_features, NOMINAL_COLUMNS



# Rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
MODELS_PATH = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_PATH / "churn_model.pkl"



# Configuración
RANDOM_STATE = 42
TEST_SIZE = 0.20
SELECTED_THRESHOLD = 0.315



# Carga y preparación de los datos
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Carga el dataset original y prepara las variables predictoras y target."""

    df = pd.read_csv(DATA_PATH)

    y = df["Churn"].map({
        "No": 0,
        "Yes": 1,
    })

    X = df.drop(columns="Churn")

    X = clean_features(X)

    return X, y


# Construcción del pipeline
def build_pipeline() -> Pipeline:
    """Construye el pipeline de preprocesamiento y clasificación."""

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=int,
                ),
                NOMINAL_COLUMNS,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=4,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline



# Entrenamiento

def train_model() -> None:
    """Entrena el modelo final y guarda el artefacto para inferencia."""

    print("Cargando datos...")
    X, y = load_data()

    # Reproducimos el mismo split utilizado durante el desarrollo del modelo.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Observaciones de entrenamiento: {len(X_train)}")
    print(f"Observaciones de test: {len(X_test)}")

    pipeline = build_pipeline()

    print("Entrenando modelo...")
    pipeline.fit(X_train, y_train)

    # Creamos el directorio si todavía no existe.
    MODELS_PATH.mkdir(parents=True, exist_ok=True)

    # Guardamos el pipeline entrenado y el threshold seleccionado.
    artifact = {
        "model": pipeline,
        "threshold": SELECTED_THRESHOLD,
    }

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(artifact, file)

    print("Entrenamiento completado.")
    print(f"Modelo guardado en: {MODEL_PATH}")


# Ejecución

if __name__ == "__main__":
    train_model()