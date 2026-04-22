# Laboratorio 01 — Clasificación Multiclase del Deterioro Cognitivo

Este repositorio contiene el código y los materiales desarrollados para el
**Laboratorio 01** del curso de *Deep Learning / Machine Learning*, cuyo objetivo
es construir y evaluar modelos de clasificación multiclase para predecir el nivel
de deterioro cognitivo de pacientes, medido mediante la escala **GDS**
(*Global Deterioration Scale*) y cinco recodificaciones alternativas de la misma.

---

## Tabla de Contenidos

1. [Descripción del Proyecto](#1-descripción-del-proyecto)
2. [Estructura del Repositorio](#2-estructura-del-repositorio)
3. [Requisitos Previos](#3-requisitos-previos)
4. [Instalación del Entorno](#4-instalación-del-entorno)
5. [Cómo Ejecutar el Proyecto](#5-cómo-ejecutar-el-proyecto)
6. [Pipeline del Modelo](#6-pipeline-del-modelo)
7. [Resultados Generados](#7-resultados-generados)

---

## 1. Descripción del Proyecto

El conjunto de datos contiene respuestas de **1 119 pacientes** a un cuestionario
de orientación cognitiva (15 atributos categóricos). La variable objetivo, la escala
GDS, indica el grado de deterioro cognitivo del paciente en una escala de 1 a 7.

Las principales decisiones de diseño del pipeline son:

- **Selección de características** mediante la prueba estadística χ² de Pearson
  (*Chi-Cuadrado*), aplicada sobre los datos originales antes de cualquier
  remuestreo, para evitar sesgos metodológicos.
- **Balanceo de clases** mediante SMOTE (*Synthetic Minority Over-sampling
  Technique*), aplicado exclusivamente sobre el conjunto de entrenamiento de cada
  fold de validación cruzada.
- **Validación cruzada estratificada** con `k = 5` folds, que garantiza la
  representación proporcional de cada clase en cada partición.
- **Tres familias de modelos** comparados entre sí: Bagging (Random Forest),
  Boosting (Gradient Boosting) y Stacking.

---

## 2. Estructura del Repositorio

```
laboratorio01_dl/
│
├── data/
│   ├── raw/                        # Datos originales (no modificar)
│   │   └── 15 atributos R0-R5.sav
│   └── processed/                  # Datasets procesados (generados al ejecutar)
│
├── notebooks/
│   └── exploracion.ipynb           # Análisis Exploratorio de Datos (EDA)
│
├── outputs/                        # Gráficos generados al ejecutar el pipeline
│
├── src/                            # Código fuente del pipeline
│   ├── config.py                   # Constantes y rutas del proyecto
│   ├── data_loader.py              # Carga del archivo .sav
│   ├── preprocessing.py            # Preparación y guardado de datos procesados
│   ├── feature_selection.py        # Selección de características con Chi-Cuadrado
│   ├── balancing.py                # Balanceo de clases con SMOTE
│   ├── bagging_model.py            # Modelo Bagging (Random Forest)
│   ├── boosting_model.py           # Modelo Boosting (Gradient Boosting)
│   ├── stacking_model.py           # Modelo Stacking (RF + GB + Regresión Logística)
│   ├── evaluation.py               # Validación cruzada y métricas
│   ├── visualization.py            # Generación de gráficos comparativos
│   └── main.py                     # Punto de entrada principal
│
├── environment.yml                 # Definición del entorno de Conda
└── README.md                       # Este archivo
```

---

## 3. Requisitos Previos

Antes de comenzar, asegúrese de tener instalado en su equipo:

- **Conda** (se recomienda [Miniforge](https://github.com/conda-forge/miniforge/releases)
  o [Miniconda](https://docs.conda.io/en/latest/miniconda.html)).
  Conda es el gestor de entornos que instalará automáticamente Python y todas las
  bibliotecas necesarias en una carpeta aislada, sin afectar el resto del sistema.

> **¿Qué es Conda?** Es una herramienta que permite crear "entornos virtuales",
> es decir, espacios aislados donde se instalan las versiones exactas de Python y
> de cada biblioteca que el proyecto necesita. De esta forma, el proyecto siempre
> funciona de la misma manera, independientemente del sistema operativo o de otros
> programas instalados.

---

## 4. Instalación del Entorno

Abra una terminal en la carpeta raíz del proyecto y ejecute los siguientes
comandos **en orden**:

### Paso 1 — Crear el entorno virtual

```bash
conda env create -f environment.yml
```

Este comando lee el archivo `environment.yml` e instala Python 3.10 junto con
todas las bibliotecas requeridas (pandas, scikit-learn, matplotlib, etc.).
El proceso puede tardar unos minutos la primera vez.

### Paso 2 — Activar el entorno

```bash
conda activate lab01_dl
```

A partir de este momento, la terminal estará usando el entorno del proyecto.
Sabrá que el entorno está activo porque verá `(lab01_dl)` al inicio de la línea
de su terminal.

> **Importante:** debe activar el entorno **cada vez** que abra una nueva terminal
> antes de ejecutar cualquier comando del proyecto.

### Paso 3 — Instalar dependencias adicionales

El archivo `environment.yml` no incluye `imbalanced-learn` (biblioteca de SMOTE)
ni `tabulate` (para los reportes en formato tabla). Instálelas con:

```bash
pip install imbalanced-learn tabulate pyreadstat
```

---

## 5. Cómo Ejecutar el Proyecto

### Opción A — Pipeline completo de modelos

Situado en la carpeta `src/`, ejecute:

```bash
cd src
python main.py
```

El programa realizará las siguientes acciones automáticamente:

1. Cargará el dataset original desde `data/raw/`.
2. Guardará una versión procesada de los datos en `data/processed/` por cada
   variable objetivo.
3. Entrenará y evaluará los tres modelos (Bagging, Boosting y Stacking) para
   cada una de las seis codificaciones GDS, mediante validación cruzada
   estratificada.
4. Imprimirá en pantalla las métricas de desempeño por modelo y variable objetivo.
5. Generará dos gráficos comparativos en la carpeta `outputs/`.

### Opción B — Análisis Exploratorio de Datos (notebook)

Si prefiere explorar el análisis exploratorio antes de ejecutar los modelos,
abra Jupyter con el siguiente comando (desde la raíz del proyecto):

```bash
jupyter notebook notebooks/exploracion.ipynb
```

El notebook ya se encuentra pre-ejecutado con sus salidas guardadas, por lo que
puede revisarlo sin necesidad de ejecutar ninguna celda adicional. Si desea
regenerar las salidas, utilice la opción *"Run All"* desde el menú *Cell*.

---

## 6. Pipeline del Modelo

El siguiente diagrama resume el flujo completo de datos para cada variable objetivo:

```
Dataset original (.sav)
        │
        ▼
  Carga y limpieza          ← data_loader.py / preprocessing.py
        │
        ├──► Guardado en data/processed/   (CSV por variable objetivo)
        │
        ▼
Validación cruzada (k=5)    ← evaluation.py
        │
   ┌────┴────────────────────────────┐
   │  Por cada fold de entrenamiento │
   │                                 │
   │  1. Selección χ² (top-10)       │ ← feature_selection.py
   │  2. Balanceo con SMOTE          │ ← balancing.py
   │  3. Entrenamiento del modelo    │ ← bagging / boosting / stacking
   │  4. Predicción sobre test fold  │
   └─────────────────────────────────┘
        │
        ▼
  Cálculo de métricas       ← evaluation.py
  (Accuracy, Precision, Recall, F1-macro)
        │
        ▼
  Gráficos comparativos     ← visualization.py → outputs/
```

---

## 7. Resultados Generados

Al finalizar la ejecución de `main.py`, encontrará:

| Artefacto | Ubicación | Descripción |
|---|---|---|
| Datasets procesados | `data/processed/processed_GDS*.csv` | Un archivo CSV por variable objetivo con los atributos predictivos y la etiqueta correspondiente |
| Gráfico de Accuracy | `outputs/accuracy_comparison.png` | Comparación de accuracy por modelo y variable objetivo |
| Gráfico de F1-Score | `outputs/f1_comparison.png` | Comparación de F1-Score (macro) por modelo y variable objetivo |
| Reporte en consola | Salida estándar | Tablas detalladas de métricas por modelo y variable objetivo |
