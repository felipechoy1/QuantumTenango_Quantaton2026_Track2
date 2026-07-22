# QuantumTenango - Quantathon 2026 Track 2

Entorno de trabajo para el notebook `Código_Hackaton_v1.ipynb` (QSVM con Qiskit).

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
- jupyterlab, ipykernel

## Historial de cambios al entorno

- 2026-07-22: se añadió `pennylane` (faltaba al ejecutar el notebook, `ModuleNotFoundError`). Se revisaron todos los imports del notebook y no falta ninguna otra dependencia.
