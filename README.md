# Customer Churn Prediction

Proyecto end-to-end de Machine Learning para predecir el abandono de clientes (*customer churn*) en una empresa de telecomunicaciones.

El proyecto cubre el ciclo completo de desarrollo de una solución de Machine Learning: análisis exploratorio, preprocesamiento, comparación y selección de modelos, ajuste de hiperparámetros, evaluación, definición de un umbral de decisión orientado al negocio, serialización del modelo y despliegue mediante una API REST con FastAPI y Docker.

---

## 1. Problema de negocio

La pérdida de clientes representa un problema relevante para empresas que trabajan con modelos de suscripción.

El objetivo de este proyecto es construir un modelo capaz de estimar la **probabilidad de que un cliente abandone el servicio**, permitiendo identificar clientes con mayor riesgo de churn y priorizar potenciales acciones de retención.

El problema se plantea como una tarea de **clasificación binaria**:

- `0` → el cliente no abandona el servicio.
- `1` → el cliente abandona el servicio.

Además de obtener buenas probabilidades, el proyecto presta especial atención al **recall de la clase churn**, ya que se parte del supuesto de negocio de que no detectar a un cliente que va a abandonar puede ser más costoso que contactar a un cliente que finalmente no abandona.

> Este supuesto permite orientar la selección del umbral, pero no representa una optimización económica real, ya que no se dispone de costes de retención o pérdida de cliente.

---

## 2. Dataset

Se utiliza el dataset **IBM Telco Customer Churn**, que contiene información sobre clientes de una empresa de telecomunicaciones.

El conjunto de datos contiene:

- **7.043 clientes**
- **21 variables originales**
- Información demográfica
- Servicios contratados
- Tipo de contrato
- Método de pago
- Antigüedad del cliente
- Cargos mensuales y acumulados
- Variable objetivo `Churn`

Distribución de la variable objetivo:

| Clase | Clientes | Porcentaje |
|---|---:|---:|
| No Churn | 5.174 | 73,46 % |
| Churn | 1.869 | 26,54 % |

Existe, por tanto, cierto **desbalance de clases**, lo que hace especialmente relevante utilizar métricas adicionales a la accuracy.

---

## 3. Flujo del proyecto

El desarrollo se divide en cuatro notebooks principales:

### `01_eda.ipynb` — Análisis exploratorio

Análisis inicial del dataset para comprender:

- estructura de los datos;
- distribución de la variable objetivo;
- tipos de variables;
- valores ausentes;
- variables categóricas y numéricas;
- relaciones entre características y churn.

### `02_preprocessing.ipynb` — Preprocesamiento

Preparación de los datos para Machine Learning:

- conversión de `TotalCharges` a formato numérico;
- tratamiento de valores ausentes;
- eliminación de `customerID`;
- codificación de variables binarias;
- One-Hot Encoding de variables nominales;
- separación estratificada entre entrenamiento y test.

El conjunto se divide en:

- **5.634 observaciones de entrenamiento**
- **1.409 observaciones de test**

utilizando:

```python
random_state = 42
test_size = 0.20
stratify = y
```

El `OneHotEncoder` se ajusta exclusivamente utilizando los datos de entrenamiento para evitar **data leakage**.

### `03_model_training.ipynb` — Entrenamiento

Se comparan diferentes algoritmos de clasificación:

- Logistic Regression
- Random Forest
- XGBoost

La evaluación durante el entrenamiento utiliza **Stratified K-Fold Cross Validation con 5 folds**.

Debido al desbalance de clases, la métrica principal utilizada para seleccionar el modelo es **Average Precision**, complementada con precision, recall y F1-score.

Tras el ajuste de hiperparámetros, XGBoost obtiene el mejor Average Precision dentro del criterio establecido, aunque las diferencias entre los modelos evaluados son reducidas.

### `04_model_evaluation.ipynb` — Evaluación final

El modelo seleccionado se evalúa sobre el conjunto de test, que se mantiene separado durante el proceso de entrenamiento y selección.

También se analiza el impacto del umbral de clasificación sobre precision y recall.

---

## 4. Modelo final

El modelo seleccionado es **XGBoost (`XGBClassifier`)**.

Los principales hiperparámetros utilizados son:

```python
XGBClassifier(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=4,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.7,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
)
```

El modelo se integra en un `Pipeline` de Scikit-learn junto con el preprocesamiento categórico, permitiendo conservar conjuntamente las transformaciones aprendidas y el clasificador.

---

## 5. Resultados

Resultados obtenidos sobre el conjunto de test:

| Métrica | Resultado |
|---|---:|
| ROC-AUC | **0,8494** |
| Average Precision | **0,6670** |
| Precision @ 0,315 | **0,5379** |
| Recall @ 0,315 | **0,7594** |
| F1-score @ 0,315 | **0,6297** |

El ROC-AUC y Average Precision se calculan utilizando las probabilidades generadas por el modelo, mientras que precision, recall y F1 dependen del umbral de clasificación seleccionado.

---

## 6. Selección del threshold

Utilizar automáticamente un threshold de `0.5` no siempre es la decisión más adecuada desde el punto de vista de negocio.

Con el threshold estándar de `0.5`, el modelo obtiene:

| | Predicción No Churn | Predicción Churn |
|---|---:|---:|
| **No Churn real** | 936 | 99 |
| **Churn real** | 176 | 198 |

Esto supone un recall para churn de aproximadamente **52,94 %**.

Para este proyecto se establece como objetivo aproximado alcanzar un **75 % de recall** sobre churn.

El threshold se selecciona utilizando probabilidades *out-of-fold* generadas exclusivamente a partir del conjunto de entrenamiento, evitando utilizar el conjunto de test para ajustar esta decisión.

El threshold seleccionado es:

```text
0.315
```

Con este umbral, la matriz de confusión sobre test es:

| | Predicción No Churn | Predicción Churn |
|---|---:|---:|
| **No Churn real** | 791 | 244 |
| **Churn real** | 90 | 284 |

Comparado con el threshold de `0.5`:

- se detectan **86 clientes churn adicionales**;
- los falsos negativos disminuyen de **176 a 90**;
- los falsos positivos aumentan de **99 a 244**.

El resultado refleja un trade-off deliberado: aumentar la capacidad de detectar clientes con riesgo de abandono a costa de generar más falsos positivos.

Este threshold **no debe interpretarse como económicamente óptimo**. Para determinar un umbral óptimo de negocio sería necesario disponer, entre otros factores, del coste de una acción de retención y del coste esperado de perder un cliente.

---

## 7. Entrenamiento

El entrenamiento se ha extraído de los notebooks a código reutilizable.

Una vez disponible el dataset en:

```text
data/raw/Telco-Customer-Churn.csv
```

el modelo puede entrenarse ejecutando desde la raíz del proyecto:

```bash
python -m src.train
```

El proceso:

1. carga los datos originales;
2. aplica las transformaciones deterministas;
3. realiza la separación train/test;
4. ajusta el preprocesamiento;
5. entrena XGBoost;
6. serializa el pipeline y el threshold.

El artefacto generado se almacena en:

```text
models/churn_model.pkl
```

El archivo contiene:

```text
churn_model.pkl
├── model
│   ├── preprocesamiento ajustado
│   └── XGBoost entrenado
│
└── threshold = 0.315
```

El modelo se serializa utilizando `pickle`.

---

## 8. Inferencia

La lógica de inferencia se encuentra en:

```text
src/predict.py
```

Una predicción sigue el flujo:

```text
Cliente
   ↓
clean_features()
   ↓
Pipeline entrenado
   ↓
predict_proba()
   ↓
Probabilidad de churn
   ↓
Threshold
   ↓
Predicción
```

Ejemplo de respuesta:

```json
{
  "churn_probability": 0.7338833808898926,
  "churn_prediction": 1,
  "threshold": 0.315
}
```

---

## 9. API REST

El modelo se expone mediante **FastAPI**.

Para ejecutar la API localmente:

```bash
python -m uvicorn api.main:app --reload
```

La API estará disponible en:

```text
http://127.0.0.1:8000
```

FastAPI genera automáticamente documentación interactiva mediante Swagger UI:

```text
http://127.0.0.1:8000/docs
```

### Endpoint de predicción

```text
POST /predict
```

Ejemplo de petición:

```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 5,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 95.0,
  "TotalCharges": 475.0
}
```

Ejemplo de respuesta:

```json
{
  "churn_probability": 0.7338833808898926,
  "churn_prediction": 1,
  "threshold": 0.315
}
```

---

## 10. Docker

La API puede ejecutarse dentro de un contenedor Docker para disponer de un entorno reproducible e independiente de la configuración local.

Antes de construir la imagen debe existir el modelo entrenado:

```bash
python -m src.train
```

### Construir la imagen

Desde la raíz del proyecto:

```bash
docker build -t customer-churn-api .
```

La imagen contiene:

- Python 3.12;
- dependencias necesarias;
- código de `api/`;
- código de `src/`;
- modelo entrenado.

### Ejecutar el contenedor

```bash
docker run --name churn-api -p 8000:8000 customer-churn-api
```

El mapeo:

```text
8000:8000
```

conecta el puerto `8000` de la máquina host con el puerto `8000` del contenedor.

Una vez iniciado:

```text
http://127.0.0.1:8000/docs
```

permite probar la API desde Swagger UI.

El servidor Uvicorn se configura dentro del contenedor utilizando:

```text
--host 0.0.0.0
```

para permitir conexiones externas al proceso que se ejecuta dentro del contenedor.

---

## 11. Reproducibilidad

Flujo general para reproducir el proyecto:

```bash
# Clonar el repositorio
git clone <URL_DEL_REPOSITORIO>

cd customer-churn-prediction

# Crear entorno virtual
uv venv

# Activarlo
source .venv/bin/activate

# Instalar dependencias
uv pip sync requirements.txt
```

Después, colocar el dataset original en:

```text
data/raw/Telco-Customer-Churn.csv
```

Entrenar el modelo:

```bash
python -m src.train
```

Ejecutar la API:

```bash
python -m uvicorn api.main:app --reload
```

O construir y ejecutar la aplicación mediante Docker:

```bash
docker build -t customer-churn-api .
docker run --name churn-api -p 8000:8000 customer-churn-api
```

---

## 15. Decisiones técnicas

Algunas de las principales decisiones tomadas durante el desarrollo son:

**Separación entre experimentación y producción.**  
Los notebooks se utilizan para análisis, experimentación y evaluación, mientras que la lógica necesaria para entrenamiento e inferencia se encuentra en módulos reutilizables dentro de `src/`.

**Prevención de data leakage.**  
Las transformaciones que aprenden parámetros de los datos se ajustan únicamente utilizando el conjunto de entrenamiento.

**Average Precision como métrica de selección.**  
Debido al desbalance de la variable objetivo, la selección del modelo no se basa únicamente en accuracy.

**Threshold orientado a recall.**  
El threshold se selecciona utilizando predicciones out-of-fold sobre entrenamiento y no utilizando el conjunto de test.

**Test reservado para evaluación final.**  
El conjunto de test se utiliza para estimar el comportamiento final del modelo después de completar el proceso de selección.

**Separación entre API y Machine Learning.**  
FastAPI consume la lógica definida en `src/`, evitando duplicar el preprocesamiento o la inferencia dentro de los endpoints.

**Modelo fuera del control de versiones.**  
El archivo `.pkl` se genera a partir del código de entrenamiento y no se almacena directamente en Git.

## Conclusión final

Este proyecto muestra el desarrollo de una solución de Machine Learning más allá del entrenamiento de un modelo en un notebook.

Partiendo de un problema de churn, se construye un flujo reproducible que incluye análisis de datos, preprocesamiento, validación cruzada, comparación de modelos, ajuste de hiperparámetros, evaluación sobre un conjunto de test independiente y selección de un threshold acorde con un objetivo de negocio.

Finalmente, el modelo se transforma en un artefacto reutilizable, se expone mediante una API REST y se empaqueta con Docker, completando un flujo **end-to-end desde los datos hasta la inferencia**.
