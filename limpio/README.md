# QuantumTenango - Quantathon 2026 Track 2

Versión limpia y unificada del proyecto: **un solo cuaderno** (`qsvm_water.ipynb`)
que reproduce todos los resultados de punta a punta con un *run all*, y **un solo
archivo de funciones** (`funciones.py`). Esta carpeta (`limpio/`) es el resultado
del proceso de migración; al terminar, `main` quedará con este contenido y lo
demás pasará a una carpeta `anterior/` sin seguimiento.

## Estructura

- `qsvm_water.ipynb`: cuaderno único de flujo de trabajo (ver "Pasos del cuaderno").
- `funciones.py`: todas las funciones del proyecto (análisis, SVM clásico,
  kernel cuántico, resultados). Ninguna lógica reutilizable vive en celdas del
  cuaderno.
- `requirements.txt`: dependencias exactas del proyecto.
- `data/raw/`: dataset original (`water_potability.csv`).
- `data/processed/dataset_v1.xlsx`: datasets generados por el Paso 1 (hojas
  `originales`, `imputados`, `normalizados`, `muestreos`, `variables`).
- `data/runs/`: matrices kernel persistidas por familia
  (`{Familia}_Train_{tamaño}.csv` / `{Familia}_Test_{tamaño}.csv`, separador `;`).
- `data/img/`: figuras del cuaderno, con convención
  `{paso:02d}_{sección:02d}_{descripción}.png` (ej. `01_02_metricas_forward.png`,
  `03_04_resultado_zyy_32.png`).
- `cache/`: resultados persistidos (JSON por hash) de `evaluar_forward` y
  `evaluar_grid_search`; evita reentrenar lo que ya se ejecutó. En esta carpeta
  sí se versiona (documenta la ejecución del análisis).

## Pasos del cuaderno

### Paso 1. Preparación de baseline (SVM clásico)
Carga y análisis del dataset, partición estratificada, imputación por medianas
de train, estandarización, selección de variables por forward selection
(`evaluar_forward`, con caché), optimización de hiperparámetros
(`evaluar_grid_search`, con caché), modelo final y guardado de
`dataset_v1.xlsx` (datasets + muestreo jerárquico balanceado + variables
seleccionadas). Toda fuente de aleatoriedad lleva semilla explícita
(`random_state=42`) para replicabilidad completa.

### Paso 2. Kernel cuántico
- **2.1 Familias**: `FAMILIAS` define a mano las combinaciones
  `(feature_map, tamaño)` — feature maps `zz`, `zyy`, `ry_cx_rx`; tamaños 16 y
  32 — con dos rutas opcionales `(feature_map, tamaño, ruta_train, ruta_test)`
  hacia matrices ya calculadas (idempotencia: lo ya calculado no se recalcula;
  `""` marca la matriz que falte).
- **2.2 Datos**: tabla resumen de registros train/test por familia (según el
  nivel de muestreo: 16→nivel 3, 32→nivel 2) y dimensión cuántica (un qubit
  por variable seleccionada).
- **2.3 Inspección**: los 6 circuitos U(x)/U(x)† de los 3 feature maps,
  apilados y navegables.
- **2.4 Matrices**: `cargar_ktrain`/`cargar_ktest` cargan lo ya persistido
  (tabla de estado, sin interacción, apto para *run all*). Las familias
  pendientes se calculan con `calcular_ktrain` (2.4.1) o `calcular_ktest`
  (2.4.2): backend local síncrono, o job de Nexus + `consultar_ktrain`/
  `consultar_ktest`. Backend por defecto: `local_selene_statevector`; también
  se admiten simuladores y hardware vía Nexus (`MATRIX_BACKEND_OPTIONS`).

### Paso 3. Resultados por familia
Con las matrices persistidas: SVM de kernel precomputado por familia (búsqueda
de `C` por validación cruzada), comparación de métricas train vs test (brecha
de overfitting), figura 1×3 por familia (heatmap de K_train + matrices de
confusión de train y test, guardada en `data/img/`) y tabla resumen de las 6
familias ordenada por F1 de test.

## Entorno Conda

Entorno de validación de la migración: `qsvm_migration` (Python 3.11). Para
recrearlo en cualquier máquina:

```bash
# 1. Crear el entorno con Python 3.11
conda create -n qsvm_migration python=3.11 -y

# 2. Activar el entorno
conda activate qsvm_migration

# 3. Instalar las dependencias del proyecto
pip install -r requirements.txt

# 4. Registrar el entorno como kernel de Jupyter
python -m ipykernel install --user --name qsvm_migration --display-name "qsvm_migration"

# 5. Abrir el cuaderno (selecciona el kernel "qsvm_migration")
jupyter lab
```

Para eliminarlo si algo sale mal:

```bash
conda deactivate
conda env remove -n qsvm_migration
```

## Dependencias principales

- numpy, pandas, scikit-learn, imbalanced-learn
- guppylang, qnexus, pytket, selene-sim (kernel cuántico: circuitos pytket,
  ejecución local Selene y jobs en Quantinuum Nexus)
- qiskit, qiskit-aer, pennylane (disponibles en el entorno; el flujo limpio
  usa la vía pytket/guppy)
- matplotlib, seaborn (visualización)
- tqdm (barras de progreso de texto; sin dependencias de frontend)
- openpyxl (lectura/escritura de `dataset_v1.xlsx`)
- jupyterlab, ipykernel

Pins exactos: `tket-exts==0.12.4` y `hugr==0.16.0` (versiones posteriores
rompen `guppy.load_pytket`; no actualizar sin validar ese flujo).

## Convenciones del proyecto

- **Un solo cuaderno, un solo archivo de funciones**: toda función nueva va a
  `funciones.py`, nunca queda suelta en una celda.
- **Semillas explícitas**: cualquier componente con aleatoriedad declara su
  `random_state`/semilla, aunque el default coincida.
- **Figuras**: se guardan siempre en `data/img/` con la convención
  `{paso:02d}_{sección:02d}_{descripción}.png`.
- **Idempotencia de matrices kernel**: las rutas en `FAMILIAS` son la fuente de
  verdad de lo ya calculado; un *run all* carga lo persistido y nunca dispara
  cómputo cuántico por sí solo (el cálculo es siempre una llamada explícita).
- **Sin dependencias de frontend**: nada de ipywidgets; tablas de estado como
  DataFrames y barras de progreso de texto, para que el cuaderno funcione igual
  en cualquier entorno (Jupyter, VS Code, etc.).

## Historial de la migración

- 2026-07-24: copia de `Código_Hackaton_v1.ipynb` como `qsvm_water.ipynb` con
  imports podados; `funciones.py` propio; guardado de figuras en `data/img`;
  semillas explícitas; secciones numeradas bajo "Paso 1".
- 2026-07-24: Paso 2 — familias de modelos con rutas de idempotencia, tabla de
  datos por familia, inspección de feature maps, y migración del flujo de
  matrices kernel (`iniciar_matriz_kernel`, `iniciar_matriz_kernel_test`,
  `consultar_matriz_nexus`, `guardar_kernel_qsvm`, etc.) desde
  `funciones_nexus.py`.
- 2026-07-24: `evaluar_grid_search` con caché en disco (igual que
  `evaluar_forward`); 2.4 reescrito sin ipywidgets (frágil en VS Code) a
  funciones planas con tabla de estado; 2.4.1/2.4.2 opcionales para calcular
  K_train/K_test pendientes; Paso 3 de resultados por familia (referencia:
  `analisis_familias_kernel.ipynb`).
