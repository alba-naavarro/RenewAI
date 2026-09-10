# 🪴 RenewAI
### Análisis y modelización predictiva de la generación solar y eólica en España

El proyecto analiza la generación solar y eólica en España y desarrolla modelos de Machine Learning capaces de estimar la generación de ambas tecnologías a partir de información meteorológica e histórica. Los resultados se integran posteriormente con datos de demanda eléctrica para analizar la cobertura solar-eólica y su relación con el precio mayorista de la electricidad.

Como parte final del proyecto se ha desarrollado también una aplicación interactiva en **Streamlit**, que permite explorar los resultados obtenidos y utilizar un simulador What-if para analizar escenarios meteorológicos hipotéticos.

---

## 🎯 Objetivos del proyecto

El objetivo principal de RenewAI es desarrollar una solución de análisis y modelización predictiva de la generación renovable en España, centrándose en la energía solar y eólica terrestre.

Para ello, el proyecto incluye:

- Integración de diferentes fuentes de datos energéticos y meteorológicos.
- Análisis exploratorio de la generación solar y eólica y de sus principales factores meteorológicos.
- Ingeniería y selección de variables relevantes para la predicción.
- Comparación de diferentes algoritmos de Machine Learning.
- Desarrollo y evaluación de modelos específicos para generación solar y eólica.
- Interpretación de los modelos mediante SHAP.
- Análisis de la cobertura de la demanda eléctrica mediante generación solar y eólica.
- Estudio de la relación entre dicha cobertura y el precio mayorista de la electricidad.
- Desarrollo de una aplicación interactiva para visualizar y explorar los resultados.

---

## 📊 Datos utilizados

El análisis utiliza datos horarios correspondientes al periodo comprendido entre **enero de 2023 y junio de 2026**.

Las principales fuentes de información son:

- **ENTSO-E Transparency Platform**: generación solar, generación eólica terrestre y demanda eléctrica.
- **Open-Meteo**: variables meteorológicas históricas para distintas localizaciones españolas.
- **Copernicus CAMS**: datos de irradiancia solar (GHI).
- **Ember**: datos históricos del precio mayorista de la electricidad en España.

Para representar las diferencias meteorológicas existentes dentro del territorio español se utilizan datos de diez localizaciones: **Valencia, Cádiz, Madrid, Bilbao, Cáceres, Santa Cruz de Tenerife, Zaragoza, A Coruña, Pamplona y Valladolid**.

---

## 🤖 Modelización

El proyecto desarrolla dos modelos predictivos independientes:

- ☀️ **Modelo solar**, destinado a estimar la generación solar agregada en España.
- 🌬️ **Modelo eólico**, destinado a estimar la generación eólica terrestre agregada en España.

Se comparan diferentes enfoques, incluyendo un baseline basado en persistencia, regresión lineal, Random Forest, XGBoost y Gradient Boosting.

Tras el proceso de comparación, selección de variables y optimización, los modelos finales utilizan **Gradient Boosting**, con conjuntos de predictores específicos para cada tecnología.

La evaluación se realiza mediante una división temporal de los datos en entrenamiento, validación y test, manteniendo el conjunto de test separado hasta la evaluación final.

---

## 🔍 Interpretabilidad y análisis de resultados

Además de evaluar la precisión predictiva mediante métricas como **MAE, RMSE y R²**, RenewAI incorpora técnicas de interpretabilidad mediante **SHAP**.

Esto permite estudiar qué variables tienen una mayor influencia sobre las predicciones y comprobar si las relaciones aprendidas por los modelos son coherentes con los patrones observados durante el análisis exploratorio.

Los resultados de generación solar y eólica se utilizan también para estimar la **cobertura solar-eólica de la demanda eléctrica**, permitiendo analizar el peso conjunto de ambas tecnologías dentro del sistema eléctrico español.

---

## 💻 Aplicación RenewAI

El proyecto incluye una aplicación interactiva desarrollada con **Streamlit** que permite consultar y explorar los principales resultados de una forma más visual.

La aplicación incluye:

- 📊 **Panel principal** con indicadores de generación, demanda y cobertura renovable.
- 🗺️ Visualización de las condiciones meteorológicas utilizadas por los modelos.
- 📈 **Modelos y resultados**, con información sobre el rendimiento predictivo.
- 💼 **Impacto y alcance**, orientado a interpretar la utilidad de los resultados.
- 🔬 **Simulador What-if**, que permite modificar variables meteorológicas y analizar cómo cambiarían las estimaciones de generación y cobertura en escenarios hipotéticos.

---

## 📁 Estructura del repositorio

```text
RenewAI/
│
├── 📓 RenewAI.ipynb
│   └── Notebook principal del proyecto
│       ├── Obtención de datos
│       ├── Integración y limpieza
│       ├── Análisis exploratorio
│       ├── Ingeniería de variables
│       ├── Modelización
│       ├── Evaluación
│       └── Interpretabilidad
│
├── 🐍 app.py
│   └── Aplicación interactiva desarrollada con Streamlit
│
├── 📂 Datasets/
│   └── Datos utilizados durante el desarrollo del proyecto
│
├── 📂 data/
│   └── Datos preparados para la aplicación
│       └── datos_app.csv
│
├── 📂 models/
│   └── Modelos y objetos necesarios para realizar las predicciones
│
├── 📂 assets/
│   └── Recursos utilizados por la interfaz de RenewAI
│
├── 📄 requirements.txt
│   └── Dependencias necesarias para ejecutar la aplicación
│
└── 📄 README.md
    └── Documentación del proyecto
