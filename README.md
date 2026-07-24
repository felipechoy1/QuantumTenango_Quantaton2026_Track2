# QuantumTenango — QSVM para potabilidad del agua

Repositorio reproducible del Track 2 de Quantathon 2026. El flujo completo está unificado en un cuaderno (`qsvm_water.ipynb`) y las funciones de modelización están centralizadas en `funciones.py`.

El cuaderno puede ejecutarse de principio a fin sin recalcular los kernels cuánticos: las matrices validadas están versionadas en `data/runs/` y un *run all* las carga de forma idempotente.

## Resultado principal

El baseline clásico sobre el conjunto completo usa cinco variables (`ph`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`) y un SVM RBF con `C=10`, `gamma="auto"` y clases balanceadas por peso. En el conjunto de prueba estratificado alcanza:

| Accuracy | Balanced accuracy | Precision | Recall | F1 | AUC |
|---:|---:|---:|---:|---:|---:|
| 0.6631 | 0.6323 | 0.5806 | 0.4922 | 0.5328 | 0.6928 |

Para el benchmark de muestras pequeñas, las familias QSVM se ordenan y la mejor familia de cada tamaño se selecciona por F1 de test:

| Train / test | Familia QSVM seleccionada | F1-CV train | F1 test QSVM | F1 test SVM-RBF | Delta |
|---:|---|---:|---:|---:|---:|
| 16 / 8 | `zz_16` | 0.5267 | 0.6000 | 0.6667 | -0.0667 |
| 32 / 16 | `zyy_32` | 0.5252 | 0.6250 | 0.6316 | -0.0066 |

Con estas muestras no se observa ventaja cuántica frente al SVM clásico. Los conjuntos de prueba tienen solo 8 y 16 observaciones balanceadas; por ello, estas cifras son un benchmark exploratorio y no una estimación estable del desempeño poblacional.

## Estructura

```text
.
├── qsvm_water.ipynb          # flujo completo y resultados ejecutados
├── funciones.py              # funciones clásicas, cuánticas y de evaluación
├── requirements.txt          # versiones directas validadas
├── cache/                    # resultados deterministas de búsquedas costosas
└── data/
    ├── raw/                  # dataset original
    ├── processed/            # dataset_v1.xlsx
    ├── runs/                 # matrices kernel y su documentación
    └── img/                  # figuras generadas por el cuaderno
```

`dataset_v1.xlsx` contiene cinco hojas:

- `originales`: observaciones originales con indicador train/test;
- `imputados`: medianas estimadas en train y aplicadas a train/test;
- `normalizados`: escalado estimado en train y aplicado a train/test;
- `muestreos`: subconjuntos balanceados, jerárquicos y reproducibles;
- `variables`: variables elegidas por forward selection.

## Datos

El archivo original contiene 3,276 observaciones, nueve predictores numéricos y la variable binaria `Potability`. Se usa el dataset público [Water Potability Dataset with 10 Parameters](https://www.kaggle.com/datasets/devanshibavaria/water-potability-dataset-with-10-parameteres), publicado por Devanshi Bavaria bajo licencia CC0 (dominio público).

La partición es 80/20, estratificada y con `random_state=42`. La separación se realiza antes de imputar o escalar. Las medianas y los parámetros de estandarización se ajustan únicamente con train.

## Metodología del cuaderno

### Paso 1 — baseline SVM

1. Análisis descriptivo y control de valores faltantes.
2. División train/test estratificada.
3. Imputación por medianas y estandarización sin usar información de test.
4. Forward selection con validación cruzada.
5. Búsqueda de `C` y `gamma` para el SVM RBF.
6. Evaluación final en test y generación de `dataset_v1.xlsx`.

La proporción aproximada 61/39 se trata con `class_weight="balanced"`; no se descartan observaciones del baseline. Los subconjuntos cuánticos sí se balancean para controlar el tamaño del experimento.

La evaluación final del método fordward define un modelo de 5 variables que son: 
ph, Hardness, Solids, Chloramines y Sulfate.

### Paso 2 — kernels cuánticos

Se evalúan tres feature maps (`zz`, `zyy`, `ry_cx_rx`) con cinco qubits —uno por variable— y tamaños de train 16 y 32. Cada familia tiene una matriz `K_train` cuadrada y una matriz `K_test` rectangular.

La configuración declarada es simulación local con el  "local_selene_statevector", 1,000 disparos, semilla 42 y diagonal unitaria sin ejecuciones redundantes. El detalle de trazabilidad está en [`data/runs/README.md`](data/runs/README.md).

El cálculo no se dispara automáticamente. `FAMILIAS` contiene las rutas de las matrices disponibles; un *run all* solo las carga. Las celdas 2.4.1 y 2.4.2 documentan cómo calcular explícitamente una familia pendiente, de forma local o en Quantinuum Nexus.

### Paso 3 — evaluación QSVM

Cada matriz se usa con `SVC(kernel="precomputed")`. El parámetro `C` se elige mediante validación cruzada estratificada de cinco pliegues en train. La tabla de familias se ordena por F1 de test.

### Paso 4 — comparación controlada

Para cada tamaño se entrena un SVM RBF clásico con exactamente las mismas
observaciones y variables que el QSVM. La mejor familia cuántica por tamaño se elige por F1 de test y se compara contra ese baseline en el mismo test.

## Instalación reproducible

Se validó con Python 3.11.15. Las versiones directas de las dependencias están fijadas en `requirements.txt`.

Clone el repositorio y entre en su raíz:

```bash
git clone https://github.com/felipechoy1/QuantumTenango_Quantaton2026_Track2.git
cd QuantumTenango_Quantaton2026_Track2
```

### Opción A — `venv` (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name qsvm-water --display-name "QSVM Water"
```

### Opción B — `venv` (macOS/Linux)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name qsvm-water --display-name "QSVM Water"
```

### Opción C — Conda

```bash
conda create -n qsvm-water python=3.11 -y
conda activate qsvm-water
python -m pip install -r requirements.txt
python -m ipykernel install --user --name qsvm-water --display-name "QSVM Water"
```

No se requieren credenciales de Nexus para reproducir los resultados versionados. Solo son necesarias si se elige un backend remoto para recalcular matrices.

## Ejecución

```bash
jupyter lab qsvm_water.ipynb
```

Seleccione el kernel `QSVM Water` y ejecute **Run All**.

## Reproducibilidad y límites

- Todas las fuentes de aleatoriedad del flujo declaran semilla 42.
- Las cachés incorporan contenido, orden y esquema de los datos, parámetros del
  modelo, configuración de CV y versión de scikit-learn.
- Test permanece aislado durante el preprocesamiento, la selección de variables y el ajuste de hiperparámetros. En el análisis QSVM sí se usa después para   ordenar las familias, de acuerdo con el criterio original del proyecto.
- El escalado se ajusta sobre todo el train antes de las validaciones cruzadas internas. Por tanto, los valores de CV sirven para selección y comparación. La evaluación final reportada se realiza en el test reservado.
- En el baseline, la selección forward y el grid de hiperparámetros se realizan sobre train; su desempeño generalizable se juzga con el test reservado.
- Los experimentos QSVM son pequeños y sensibles a cada observación y al ruido de disparos. No sustentan afirmaciones de superioridad cuántica, son útiles para explorar la implementación del algoritmo.