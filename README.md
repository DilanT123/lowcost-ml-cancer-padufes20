# 🧠 Análisis Multimodal para Clasificación de Lesiones Cutáneas (PAD-UFES-20)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Dependencies](https://img.shields.io/badge/Dependencies-Conda-orange.svg)
![Project Status](https://img.shields.io/badge/Status-Completado-brightgreen.svg)

Este repositorio contiene el código completo para un **sistema avanzado de análisis multimodal** que combina características de imágenes médicas y datos clínicos para la clasificación de lesiones cutáneas, utilizando técnicas de machine learning, reducción de dimensionalidad, selección de características y modelos de ensemble. El sistema fue desarrollado haciendo uso del **dataset PAD-UFES-20**, creado originalmente por la *Universidade Federal do Espírito Santo (UFES)*.

> ⚠️ **Nota**: Este proyecto es independiente y **no está afiliado a UFES**. El uso del dataset se realiza con fines académicos y de investigación, respetando los términos de acceso y atribución.
>
> > ✅ El script realiza: carga de datos, extracción de características, balanceo (SMOTE), entrenamiento de modelos, ensemble y guardado de resultados.
---

## 📌 Tabla de Contenidos

- [📌 Tabla de Contenidos](#-tabla-de-contenidos)
- [🎯 Objetivo](#-objetivo)
- [📦 Estructura del Proyecto](#-estructura-del-proyecto)
- [⚙️ Configuración y Requisitos](#️-configuración-y-requisitos)
- [🚀 Cómo Ejecutar](#-cómo-ejecutar)
- [📊 Resultados Clave](#-resultados-clave)
- [📈 Visualizaciones](#-visualizaciones)
- [🛠️ Funcionalidades Destacadas](#️-funcionalidades-destacadas)
- [📂 Dataset](#-dataset)
- [📄 Publicación / Post Artículo](#-publicación--post-artículo)


---

## 🎯 Objetivo

Desarrollar un modelo de aprendizaje automático robusto y explicativo para la clasificación de diagnósticos dermatológicos (como **MEL, BCC, SCC, NEV, ACK, SEK**) a partir de un enfoque **multimodal**, integrando:
- **Características de imágenes** (extraídas mediante técnicas de procesamiento)
- **Datos clínicos y demográficos**
- **Preprocesamiento avanzado**, balanceo de clases (SMOTE), selección de características y reducción de dimensionalidad (PCA)

El sistema evalúa múltiples modelos (XGBoost, Random Forest, SVM, etc.) y combina los mejores mediante **ensemble learning dinámico**.

---

## 📦 Estructura del Proyecto

```
PAD_UFES_Multimodal/
│
├── data/                     # Datos de entrada
│   ├── metadata.csv          # Información clínica de pacientes y lesiones
│   └── imagenes/             # Imágenes médicas (JPEG/PNG)
│
├── resultados/               # Salidas del análisis
│   ├── multimodal_dataset.csv    # Dataset final combinado
│   └── exploration_results.json  # Estadísticas de exploración
│
├── graficos/                 # Visualizaciones generadas
│   ├── confusion_matrix_*.png
│   ├── feature_importance_*.png
│   └── ensemble_models_analysis.png
│
├── modelos/                  # Modelos entrenados y resultados
│   ├── best_model_xgboost.pkl
│   ├── enhanced_models_results.pkl
│   └── class_weights.pkl

```

---

## ⚙️ Configuración y Requisitos

### 🐍 Entorno (Conda)

Se recomienda usar `conda` para gestionar las dependencias:

```bash
# Crear entorno
conda create -n pad_ufes_env python=3.10
conda activate pad_ufes_env

# Instalar dependencias
pip install scikit-learn xgboost imbalanced-learn matplotlib seaborn pandas numpy pillow tqdm plotly joblib
```

### 🔧 Configuración Principal (`config.py`)

```python
CONFIG = {
    'paths': {
        'metadata_file': './data/metadata.csv',
        'images_folder': './data/imagenes/',
        'output_folder': './resultados/',
        'models_folder': './modelos/',
        'plots_folder': './graficos/'
    },
    'data_params': {
        'test_size': 0.25,
        'val_size': 0.15,
        'random_state': 42
    },
    'model_params': {
        'cv_folds': 5,
        'scoring_metric': 'f1_macro',
        'hyperparameter_search': 'random'
    },
    'class_balance': {
        'apply_smote': True
    },
    'dimensionality_reduction': {
        'apply_pca': True,
        'apply_feature_selection': True
    }
}
```

---

## 🚀 Cómo Ejecutar

1. **Clonar el repositorio:**

```bash
git clone https://github.com/DilanT123/lowcost-ml-cancer-padufes20.git
cd PAD_UFES_Multimodal
```

2. **Organizar los datos:**
   - Colocar `metadata.csv` en `data/`
   - Colocar las imágenes en `data/imagenes/`

3. **Ejecutar el pipeline completo:**


> ✅ El script realiza: carga de datos, extracción de características, preprocesamiento, entrenamiento, evaluación y guardado de resultados.

---

## 📊 Resultados Clave

### ✅ **Modelo Ganador Individual: XGBoost**

- **F1-score macro (validación):** `0.6452`
- **F1-score macro (test):** `0.6086`
- **Recall MEL (validación):** `0.6250` → ✅ **Detectó melanoma en 62.5% de los casos**
- **Tiempo de entrenamiento:** `72.35 s`
- **Mejores hiperparámetros:**  
  ```python
  {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1}
  ```

### 🏆 **Mejor Modelo de Ensemble: Custom Weighted**

- **F1-score:** `0.7581`
- **Accuracy:** `0.7681`
- **AUC-ROC:** `0.9314`
- **Tipo:** Ensemble personalizado con pesos automáticos basados en F1 y accuracy
- **Mejora vs XGBoost individual:**  
  - **+1.19% en F1-score**
  - **+0.76% en accuracy**

### 📈 Comparación de Ensembles

| Ensemble           | F1-Score | Accuracy | AUC-ROC  | Tiempo (s) |
|--------------------|----------|----------|----------|------------|
| **Custom Weighted**| 0.7581   | 0.7681   | 0.9314   | 0.0010     |
| Stacking           | 0.7581   | 0.7681   | 0.9353   | 41.0087    |
| Voting Soft        | 0.7554   | 0.7594   | 0.9307   | 0.2542     |
| Voting Hard        | 0.7527   | 0.7623   | -        | 0.1298     |

> ✅ El **ensemble personalizado** logra el mejor rendimiento con **mínimo tiempo de inferencia**.

---

## 📈 Visualizaciones

El sistema genera automáticamente:

- Distribución de clases (antes/después de SMOTE)
- Matrices de confusión (por modelo y conjunto)
- Importancia de características (XGBoost)
- Comparación de modelos y ensembles
- Análisis de balance de datos y calidad

> Todos los gráficos se guardan en `./graficos/` y `./plots_enhanced/` en alta resolución.


---

## 🛠️ Funcionalidades Destacadas

✅ **Extracción automática de características de imágenes**  
✅ **Balanceo de clases con SMOTE**  
✅ **Selección de características múltiple (RFE, Varianza, Info. Mutua)**  
✅ **Modelos evaluados:** XGBoost, Random Forest, Extra Trees, SVM-RBF  
✅ **Ensemble Learning Dinámico con pesos ajustables**  
✅ **Soporte para gráficos interactivos (Plotly)**  
✅ **Logging detallado y guardado de resultados (Pickle, JSON, CSV)**  
✅ **Resumen final con métricas y tiempos de ejecución**

---

## 📂 Dataset

- **Origen:** Colección clínica de la UFES (datos anonimizados)
- **Tamaño:** 2298 muestras
- **Columnas:** 26 (20 categóricas, 3 numéricas, 2 enteras, 1 booleana)
- **Clases objetivo:** `MEL`, `BCC`, `SCC`, `NEV`, `ACK`, `SEK`
- **Imágenes:** Tamaño estándar `(224, 224)` en formato RGB

---

## 📄 Publicación / Post Artículo

Este código fue utilizado para generar los resultados del **artículo científico**. Incluye análisis reproducible, visualizaciones publicables y documentación detallada para facilitar la revisión y replicación.

> 📄 **Título del artículo:**  
> *"EVALUACIÓN COMPARATIVA DE MODELOS DE APRENDIZAJE AUTOMÁTICO DE BAJO COSTO PARA DIAGNÓSTICO DE CÉLULAS CANCEROSAS"*

---


> Desarrollado por: **Dilan Torres, Madelein Conforme**


