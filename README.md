# analisisFase3 Mauricio Salazar

# Prediction Project UNAD

E# Proyecto de Análisis y Modelado Predictivo de Datos de Vehículos, Condiciones Cardíacas y Vinos

Este proyecto tiene como objetivo aplicar modelos de machine learning sobre tres conjuntos de datos distintos: vehículos, condiciones cardíacas y características de vinos. Se realiza un análisis exploratorio de datos (EDA) y se desarrollan modelos predictivos personalizados utilizando regresión lineal, regresión logística y árboles de decisión, evaluando su precisión y aplicabilidad para cada caso de estudio.

## Descripción de Archivos de Datos

### 1. Datos de Vehículos
- **`vehicles.csv`**: Contiene datos relacionados con vehículos, como precio, kilometraje, modelo y características técnicas adicionales. Este dataset es utilizado para predecir el precio de un vehículo en función de sus características mediante un modelo de regresión lineal.

### 2. Datos de Condiciones Cardíacas
- **`heart.csv`**: Contiene información médica de pacientes, como edad, colesterol, presión arterial y presencia de alguna condición cardíaca. Este conjunto de datos es utilizado para construir un modelo de regresión logística que predice la probabilidad de que un paciente tenga una condición cardíaca.

### 3. Datos de Vinos
- **`winequality-red.csv`** y **`winequality-white.csv`**: Conjuntos de datos que contienen características fisicoquímicas de vinos tintos y blancos, incluyendo acidez, azúcares, pH, contenido de alcohol y una clasificación de calidad (`quality`). Este dataset se utiliza para clasificar la calidad del vino mediante un modelo de árbol de decisión.

## Objetivos del Proyecto

1. **Desarrollar un modelo de predicción para cada dataset**:
   - Vehículos: Estimar el precio en función de características clave usando regresión lineal.
   - Condiciones cardíacas: Predecir la presencia de una enfermedad cardíaca con regresión logística.
   - Vinos: Clasificar la calidad del vino usando un modelo de árbol de decisión.
2. **Aplicar Análisis Exploratorio de Datos (EDA)**: 
   - Examinar y visualizar cada conjunto de datos para identificar tendencias, relaciones entre variables y valores atípicos.
3. **Evaluar el rendimiento de cada modelo**:
   - Calcular métricas como Error Cuadrático Medio (MSE), precisión, recall y accuracy para evaluar los modelos.

## Implementación del Modelo

### 1. Predicción de Precio de Vehículos con Regresión Lineal

- **EDA**: Visualización de correlaciones entre precio y características como kilometraje, año, y otras variables cuantitativas.
- **Modelo**: Usamos regresión lineal para predecir el precio de los vehículos.
- **Evaluación**: Calculamos el Error Cuadrático Medio (MSE) y el Coeficiente de Determinación (R²) para medir la precisión del modelo.

Ejemplo de implementación:
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Cargar y preprocesar datos
X = vehicles_data[['mileage', 'year', 'feature_x']]
y = vehicles_data['price']

# División del conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluación
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


### 2. Predicción de Condiciones Cardíacas con Regresión Logística

- **EDA**: Exploración de variables relevantes como presión arterial, colesterol y edad.
- **Modelo**: Se implementa una regresión logística para predecir la probabilidad de enfermedad cardíaca.
- **Evaluación**: Medimos el desempeño del modelo con métricas como accuracy, precisión y recall.

Ejemplo de implementación:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Cargar y preprocesar datos
X = heart_data[['age', 'cholesterol', 'blood_pressure']]
y = heart_data['target']

# División del conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluación
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)



### 3. Clasificación de la Calidad del Vino con Árbol de Decisión

- **EDA**: Análisis de variables como acidez, nivel de azúcar, pH y alcohol, observando su influencia en la calidad del vino.
- **Modelo**: Se construye un árbol de decisión para clasificar la calidad de los vinos en función de sus características fisicoquímicas.
- **Evaluación**: Se mide la precisión del modelo y se visualiza la estructura del árbol para entender las decisiones basadas en cada característica.

Ejemplo de implementación:
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Cargar y preprocesar datos
X = wine_data[['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'pH', 'alcohol']]
y = wine_data['quality']

# División del conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluación
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Visualización del árbol de decisión
plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=X.columns)
plt.title("Árbol de Decisión para Clasificación de Calidad de Vinos")
plt.show()

## Consideraciones y Conclusiones Generales

- **Sujeto a Mejoras**: Todos los modelos desarrollados en este proyecto están sujetos a mejoras. Es fundamental explorar nuevas características y realizar un ajuste continuo de los hiperparámetros para optimizar el rendimiento.

- **Validación y Evaluación**: Se utilizaron métricas como el Error Cuadrático Medio (MSE) y el coeficiente de determinación \(R^2\) para evaluar el modelo de regresión lineal, así como la matriz de confusión y la precisión para la regresión logística. Estas métricas son clave para entender el ajuste y la efectividad de los modelos.

- **Preprocesamiento**: El preprocesamiento de datos, incluyendo la conversión de variables categóricas a dummy y el escalado de variables numéricas, es crítico para el desempeño de los modelos. Un enfoque cuidadoso en esta etapa puede mejorar significativamente los resultados.

- **Exploración de Datos**: El análisis exploratorio de datos es esencial para identificar relaciones significativas y tendencias. Esto ayuda a formular hipótesis y decisiones informadas sobre qué características incluir en los modelos.

- **Resultados**: Los resultados obtenidos de los modelos indican que hay correlaciones significativas entre las variables seleccionadas y las variables objetivo. Por ejemplo, en el análisis de vehículos, factores como el precio presente y el tipo de combustible mostraron un impacto considerable en el precio de venta.

- **Futuras Direcciones**: Se recomienda realizar pruebas adicionales con nuevas características, así como experimentar con otros algoritmos de aprendizaje automático como los árboles de decisión, redes neuronales o métodos de ensamble para mejorar la capacidad predictiva del modelo.

## Requisitos del Proyecto

Para ejecutar este proyecto, asegúrate de tener las siguientes bibliotecas instaladas:
```bash
pip install pandas numpy scikit-learn

