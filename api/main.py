"""API REST para el modelo de predicción de churn."""

from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_churn


app = FastAPI(
    title="Customer Churn Prediction API",
    description="API para predecir la probabilidad de abandono de clientes.",
    version="1.0.0",
)


class CustomerData(BaseModel):
    """Datos de entrada necesarios para realizar una predicción."""

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def root():
    """Comprueba que la API está funcionando."""

    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.post("/predict")
def predict(customer: CustomerData):
    """Genera una predicción de churn para un cliente."""

    customer_dict = customer.model_dump()

    result = predict_churn(customer_dict)

    return result