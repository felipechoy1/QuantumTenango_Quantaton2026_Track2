# QuantumTenango - Quantathon 2026 Track 2

Entorno de trabajo para el notebook `Código_Hackaton_v1.ipynb` (QSVM con Qiskit).

## Estructura de código

- `funciones.py`: funciones ya validadas y estables, usadas por el notebook clásico (gráficas de análisis de variables, `evaluar_forward`, métricas de clasificación, matrices de confusión, exportación de datasets, etc.).
- `Código_Hackaton_v1.ipynb`: notebook principal, con el análisis y la modelización SVM/QSVM.
- `funciones_nexus.py`: funciones de la parte cuántica con guppy/Nexus — conexión y jobs de Quantinuum Nexus, ejecución local en Selene, guardado de resultados, y toda la lógica del kernel ZZ (feature map en pytket, circuito `U(x_j)† U(x_i)`, puente `guppy.load_pytket`, matriz kernel por pares).
- `guppy_kernel_qsvm.ipynb`: notebook del kernel cuántico ZZ para QSVM (kernel `quantum_space`). Celdas de solo parámetros y llamadas; la lógica vive en `funciones_nexus.py`.
- `guppy_real_sim_control.ipynb`: notebook de control/consulta de jobs guppy en Nexus (kernel `quantum_space`).

## Datos generados

- `data/raw/`: dataset original (`water_potability.csv`).
- `data/processed/`: datasets exportados por `guardar_datasets_excel` (celda final del notebook). Genera un único `.xlsx` con tres hojas (`originales`, `imputados`, `normalizados`), cada una con train y test unificados y columnas de control `_PartInd_` (0=train, 1=test), `_Imputation_` (1 si la fila tuvo algún valor imputado) e `_ImputationCol_` (nombre de las columnas imputadas, separadas por coma si es más de una). Incluye también `df_escalado.csv` (features escaladas + `_PartInd_` + `Potability`, separador `;`), el insumo del kernel cuántico.
- `data/runs/`: resultados de ejecuciones cuánticas en CSV (separador `;` en todos): `run_*.csv` (conteos completos de una ejecución), `kernel_run_*.csv` (resumen compacto de un par del kernel) y `kernel_matrix_run_*.csv` (una fila por circuito de una matriz kernel, mismo esquema para runs locales y remotos).
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
- tqdm (barras de progreso en `evaluar_forward` y en la matriz kernel)
- openpyxl (exportación de datasets a Excel vía `guardar_datasets_excel`)
- jupyterlab, ipykernel, ipywidgets
- guppylang, qnexus, pytket, selene-sim (parte cuántica guppy/Nexus del kernel ZZ)

**Pins críticos**: `tket-exts==0.12.4` y `hugr==0.16.0`. Versiones mayores (tket-exts 0.14, hugr 0.18) rompen la compilación de circuitos pytket dentro de guppy (`guppy.load_pytket`). No actualizarlos sin validar ese flujo.

## Historial de cambios al entorno

- 2026-07-22: se añadió `pennylane` (faltaba al ejecutar el notebook, `ModuleNotFoundError`). Se revisaron todos los imports del notebook y no falta ninguna otra dependencia.
- 2026-07-22: se añadió `tqdm` para mostrar una barra de progreso en `evaluar_forward`.
- 2026-07-22: se añadió `openpyxl`, requerido por `guardar_datasets_excel` para escribir archivos `.xlsx`.
- 2026-07-22: `evaluar_forward` incorpora persistencia por hash (cache en disco) para evitar reentrenar si ya existe un resultado para la misma combinación de datos, modelo e hiperparámetros; imprime en consola si entrenó desde cero, cargó del cache, o reescribió el cache existente.
- 2026-07-22: `graficar_matrices_confusion` colorea por conteo (no por porcentaje), con una barra de color independiente por cada matriz (train, test) para ver la proporcionalidad de cuadrantes dentro de cada una.
- 2026-07-22: se añadieron `guppylang`, `qnexus` y `pytket` (también instalados en `qsvm`).
- 2026-07-23: homologación del kernel ZZ entregado en `guppy_kernel_handoff/`: la lógica se integró a `funciones_nexus.py`, nació `guppy_kernel_qsvm.ipynb`, y todos los CSV de runs quedaron con separador `;` y esquema único.
- 2026-07-23: se añadieron `ipywidgets` (faltaba en `qsvm`; `funciones_nexus.py` lo importa) y `selene-sim`, y se fijaron `tket-exts==0.12.4` y `hugr==0.16.0` (versiones mayores rompen `guppy.load_pytket`).

## Kernel cuántico ZZ para QSVM (`guppy_kernel_qsvm.ipynb`)

Construye el kernel de fidelidad `K(x_i, x_j) = P(00...0)` del circuito `U(x_j)† U(x_i)`, con el feature map ZZ definido en pytket y ejecutado vía guppy. Abrirlo con el kernel del entorno Conda `quantum_space`. Toda la lógica vive en `funciones_nexus.py`; el cuaderno solo tiene parámetros y llamadas.

Flujo: cargar `data/processed/df_escalado.csv` → inspeccionar el feature map y el circuito del par (sin consumir shots) → ejecutar un par de validación → construir la matriz kernel.

Puntos de operación:

- La ejecución arranca apagada (`RUN_KERNEL = False`, `RUN_MATRIX = False`); revisar circuitos y costo antes de encenderla. Con la diagonal desactivada se fija `K(i,i)=1` y para `m` filas se ejecutan `m(m-1)/2` circuitos.
- Backends de la matriz (`MATRIX_BACKEND`): `local_selene_statevector` (no toca Nexus), `nexus_selene_statevector`, `H1-1LE`/`H1-Emulator`/`H2-1LE`/`H2-Emulator` (no aceptan HUGR: se suben circuitos pytket, se compilan en Nexus y el execute job se encadena automáticamente al consultar) y `Helios-1E-lite` (HUGR directo).
- El proyecto de Nexus se configura con `PROJECT_NAME` (por defecto `base_proy`; se crea solo si no existe, vía `get_or_create`).
- Tras enviar una matriz a Nexus, **no reejecutar la celda de envío** (crearía jobs duplicados); reejecutar solo la celda de consulta hasta que aparezca `COMPLETED`. Importante: el mapeo par↔resultado vive en memoria — no reiniciar el kernel de Jupyter entre el envío y la descarga.
- Cada run queda en `data/runs/` (`kernel_run_*.csv` para pares, `kernel_matrix_run_*.csv` para matrices).

## Nota: consulta de ejecuciones en Quantinuum Nexus (`guppy_real_sim_control.ipynb`)

El notebook `guppy_real_sim_control.ipynb` permite correr una simulación guppy (local o en Selene/Nexus) o consultar jobs existentes de un proyecto. Debe abrirse siempre con el kernel del entorno Conda `quantum_space`.

- `ALLOW_NEW_EXECUTION = True`: run nuevo (local o remoto según `EXECUTION_TARGET`); no se consultan jobs existentes.
- `ALLOW_NEW_EXECUTION = False`: modo consulta; lista los jobs `EXECUTE` del proyecto con un desplegable, valida que el seleccionado esté `COMPLETED` y descarga sus resultados en una tabla (estados medidos, conteos, shots y proporciones). No se compila, sube ni envía nada nuevo.

La conexión recupera el proyecto por nombre con `get_or_create`: si el nombre no existe, se crea uno nuevo en lugar de fallar.
