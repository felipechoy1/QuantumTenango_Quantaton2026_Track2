# QuantumTenango - Quantathon 2026 Track 2

Entorno de trabajo para el notebook `Código_Hackaton_v1.ipynb` (QSVM con Qiskit).

## Estructura de código

- `funciones.py`: funciones ya validadas y estables, usadas por el notebook (gráficas de análisis de variables, `evaluar_forward`, métricas de clasificación, matrices de confusión, exportación de datasets, etc.).
- `Código_Hackaton_v1.ipynb`: notebook principal, con el análisis y la modelización SVM/QSVM.

## Datos generados

- `data/raw/`: dataset original (`water_potability.csv`).
- `data/processed/`: datasets exportados por `guardar_datasets_excel` (celda final del notebook). Genera un único `.xlsx` con tres hojas (`originales`, `imputados`, `normalizados`), cada una con train y test unificados y columnas de control `_PartInd_` (0=train, 1=test), `_Imputation_` (1 si la fila tuvo algún valor imputado) e `_ImputationCol_` (nombre de las columnas imputadas, separadas por coma si es más de una).
- `cache/`: resultados persistidos de `evaluar_forward` (no se sube al repositorio, ver `.gitignore`; se regenera automáticamente al correr el notebook).

## Entorno Conda

El entorno ya fue creado localmente con el nombre `qsvm` (Python 3.11) e instalado a partir de `requirements.txt`. Si necesitas recrearlo desde cero (otra máquina, entorno corrupto, etc.), sigue estos pasos manuales:

```bash
# 1. Crear el entorno con Python 3.11
conda create -n qsvm python=3.11 -y

# 2. Activar el entorno
conda activate qsvm

# 3. Instalar las dependencias del proyecto
pip install -r requirements.txt

# 4. Registrar el entorno como kernel de Jupyter
python -m ipykernel install --user --name qsvm --display-name "Python (qsvm)"

# 5. Abrir el notebook (selecciona el kernel "Python (qsvm)")
jupyter lab
```

Para eliminar el entorno si algo sale mal:

```bash
conda deactivate
conda env remove -n qsvm
```

## Nota sobre `venv_qsvm.rar`

Este archivo es un `venv` (entorno virtual de Python) comprimido, no un entorno Conda, y contiene rutas absolutas propias de la máquina donde se creó — no es portable entre equipos ni se puede "importar" directamente a Conda. Por eso el entorno `qsvm` se reconstruyó desde `requirements.txt` en lugar de reutilizar ese archivo. Además, por su tamaño (~293 MB) está excluido del repositorio vía `.gitignore`.

## Dependencias principales

- numpy, pandas, scikit-learn, imbalanced-learn
- qiskit, qiskit-aer (simulación cuántica)
- pennylane 0.45.1 (computación cuántica diferenciable, usada directamente en el notebook)
- matplotlib, seaborn (visualización)
- tqdm (barra de progreso en `evaluar_forward`)
- openpyxl (exportación de datasets a Excel vía `guardar_datasets_excel`)
- jupyterlab, ipykernel

## Historial de cambios al entorno

- 2026-07-22: se añadió `pennylane` (faltaba al ejecutar el notebook, `ModuleNotFoundError`). Se revisaron todos los imports del notebook y no falta ninguna otra dependencia.
- 2026-07-22: se añadió `tqdm` para mostrar una barra de progreso en `evaluar_forward`.
- 2026-07-22: se añadió `openpyxl`, requerido por `guardar_datasets_excel` para escribir archivos `.xlsx`.
- 2026-07-22: `evaluar_forward` incorpora persistencia por hash (cache en disco) para evitar reentrenar si ya existe un resultado para la misma combinación de datos, modelo e hiperparámetros; imprime en consola si entrenó desde cero, cargó del cache, o reescribió el cache existente.
- 2026-07-22: `graficar_matrices_confusion` colorea por conteo (no por porcentaje), con una barra de color independiente por cada matriz (train, test) para ver la proporcionalidad de cuadrantes dentro de cada una.

## Nota: consulta de ejecuciones en Quantinuum Nexus

El notebook `guppy_real_sim_control.ipynb` permite consultar el proyecto existente `guppy-kernel-encoding` en Quantinuum Nexus y recuperar los resultados de sus jobs de ejecución. Debe abrirse siempre con el kernel del entorno Conda `quantum_space`.

Flujo de trabajo de las primeras celdas:

1. Comprueba la autenticación con Nexus y recupera el proyecto por su nombre exacto. La consulta usa `projects.get`, por lo que no crea otro proyecto si el nombre es incorrecto.
2. Recupera los jobs de tipo `EXECUTE`, imprime su índice, nombre, estado e ID, y muestra una lista desplegable para seleccionar uno.
3. Lee el índice seleccionado, comprueba que el job esté en estado `COMPLETED`, descarga únicamente sus resultados y presenta una tabla con los estados medidos, conteos, shots y proporciones.

Por seguridad, `ALLOW_NEW_EXECUTION = False` es el valor predeterminado. En este modo se pueden consultar y descargar resultados existentes, pero no se compilan, suben ni envían nuevas ejecuciones. Después de cambiar la selección del desplegable solo hay que volver a ejecutar la tercera celda para cargar el nuevo job.
