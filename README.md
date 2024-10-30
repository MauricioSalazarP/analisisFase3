# analisisFase3 Mauricio Salazar

# Prediction Project UNAD

Este proyecto está enfocado en la clasificación de la calidad de vinos tintos a partir de características fisicoquímicas. A lo largo del proyecto, hemos implementado varios modelos de machine learning para mejorar la precisión en la predicción de la calidad del vino. Actualmente, el proyecto abarca tres tipos de modelos: Regresión Lineal, Regresión Logística y un modelo basado en Árbol de Decisión. Cada modelo está sujeto a mejoras continuas y pruebas adicionales para incluir características o técnicas de optimización.

## Tabla de Contenidos
- [Descripción del Dataset](#descripción-del-dataset)
- [Modelos Implementados](#modelos-implementados)
  - [Regresión Lineal](#regresión-lineal)
  - [Regresión Logística](#regresión-logística)
  - [Árbol de Decisión](#árbol-de-decisión)
- [Evaluación de Modelos](#evaluación-de-modelos)
- [Mejoras y Pruebas Futuras](#mejoras-y-pruebas-futuras)
- [Requisitos del Proyecto](#requisitos-del-proyecto)
- [Ejemplo de Uso](#ejemplo-de-uso)

## Descripción del Dataset

El dataset utilizado proviene de las características fisicoquímicas de vinos tintos, y contiene las siguientes columnas:
- **Características**: 
  - `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`
- **Objetivo**:
  - `quality`: una puntuación de la calidad del vino que varía de 3 a 8

## Modelos Implementados

### Regresión Lineal

Inicialmente, se implementó un modelo de Regresión Lineal para explorar la relación entre las características fisicoquímicas y la calidad del vino. Sin embargo, debido a la naturaleza discreta del objetivo (`quality`), este modelo mostró limitaciones en precisión, ya que está diseñado para variables continuas.

### Regresión Logística

El siguiente paso fue implementar una Regresión Logística, que es más adecuada para tareas de clasificación discreta. Este modelo mostró mejoras en la precisión en comparación con la regresión lineal, pero sigue presentando dificultades al clasificar algunas de las categorías de calidad con mayor precisión.

### Árbol de Decisión

Finalmente, se entrenó un modelo de Árbol de Decisión para mejorar la clasificación de la calidad del vino. El Árbol de Decisión permite capturar relaciones no lineales entre las variables de entrada, logrando así una precisión superior en algunas categorías. Este modelo, sin embargo, aún presenta margen de mejora, especialmente en la clasificación de clases minoritarias.

## Evaluación de Modelos

Cada modelo fue evaluado mediante métricas de precisión, matriz de confusión y reporte de clasificación. A continuación, se resumen los resultados de la precisión para los modelos implementados:

| Modelo              | Precisión |
|---------------------|-----------|
| Regresión Lineal    | 0.45      |
| Regresión Logística | 0.61      |
| Árbol de Decisión   | 0.61      |

El modelo de Árbol de Decisión mostró un rendimiento relativamente superior, aunque todavía es sensible al desbalance en las clases de calidad del vino.

## Mejoras y Pruebas Futuras

El proyecto se encuentra en una etapa de mejora continua. A continuación, algunas propuestas de optimización:
- **Equilibrio de clases**: Aplicar técnicas de sobremuestreo para mejorar la clasificación de clases minoritarias.
- **Prueba de otros modelos**: Implementar `RandomForestClassifier`, `GradientBoostingClassifier` o `XGBoost` para observar mejoras en la precisión.
- **Selección de características**: Identificar las características más relevantes para mejorar la precisión general.
- **Optimización de hiperparámetros**: Ajustar los parámetros de los modelos para maximizar su rendimiento.
- **Incorporación de nuevas características**: Evaluar la posibilidad de incluir nuevas variables que puedan contribuir a mejorar la precisión en la clasificación.

## Requisitos del Proyecto

Para ejecutar este proyecto, asegúrate de tener las siguientes bibliotecas instaladas:
```bash
pip install pandas numpy scikit-learn

