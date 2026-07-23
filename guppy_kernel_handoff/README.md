# Kernel ZZ con pytket, Guppy y Quantinuum Nexus

Este paquete contiene un flujo reproducible para construir un kernel cuántico
ZZ con `pytket`, inspeccionar los circuitos y ejecutarlos localmente o mediante
Quantinuum Nexus.

## Contenido

```text
guppy_kernel_handoff/
├── README.md
├── requirements-guppy.txt
├── guppy_real_sim_control.ipynb
├── funciones_nexus.py
└── data/
    ├── raw/
    │   └── df_escalado.csv
    └── runs/
```

- `guppy_real_sim_control.ipynb`: notebook principal.
- `funciones_nexus.py`: conexión, consulta, ejecución y descarga de resultados.
- `data/raw/df_escalado.csv`: variables escaladas y partición train/test.
- `data/runs/`: destino de los CSV compactos generados.
- `requirements-guppy.txt`: versiones usadas para validar el flujo.

No se incluyen credenciales, tokens de Nexus, entornos virtuales ni resultados
de ejecuciones anteriores.

## Requisitos

- Windows, Linux o macOS.
- Python 3.12 recomendado.
- Una cuenta de Quantinuum Nexus sólo para ejecuciones remotas.
- Acceso y cuota para el backend remoto seleccionado.

La ejecución `local_selene_statevector` no necesita una cuenta Nexus.

## Instalación

Desde la carpeta que contiene este README:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-guppy.txt
python -m ipykernel install --user --name guppy-kernel --display-name "Python (Guppy Kernel)"
jupyter lab
```

### Linux o macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-guppy.txt
python -m ipykernel install --user --name guppy-kernel --display-name "Python (Guppy Kernel)"
jupyter lab
```

Abre `guppy_real_sim_control.ipynb`, selecciona el kernel
`Python (Guppy Kernel)` y ejecuta las celdas en orden.

Es importante iniciar Jupyter desde esta carpeta. El notebook carga:

```python
Path("data/raw/df_escalado.csv")
```

## Flujo del notebook

1. Carga `df_escalado.csv`.
2. Separa train y test usando `_PartInd_`.
3. Construye `zz_feature_map(x)` en pytket.
4. Permite visualizar el feature map de una fila.
5. Construye `U(x_j)† U(x_i)`.
6. Permite visualizar el circuito del kernel antes de ejecutarlo.
7. Carga el circuito pytket en Guppy cuando el backend acepta HUGR.
8. Ejecuta un solo par o una matriz de filas seleccionadas.
9. Extrae únicamente el conteo de `00...0`.
10. Guarda resultados compactos en `data/runs/`.

El kernel estimado es:

```text
K(x_i, x_j) = count(00...0) / shots
```

## Inspeccionar un circuito sin ejecutarlo

Selecciona una fila para visualizar su feature map:

```python
PREVIEW_ROW = 0
```

Selecciona un par:

```python
KERNEL_ROW_I = 0
KERNEL_ROW_J = 1
```

La inspección no consume shots.

## Ejecución de una matriz

Selecciona explícitamente las filas:

```python
MATRIX_ROWS = [0, 1, 2, 3]
```

La ejecución permanece apagada mientras:

```python
RUN_MATRIX = False
```

Cuando el circuito y el costo sean aceptables:

```python
RUN_MATRIX = True
MATRIX_SHOTS = 1000
MATRIX_EXECUTE_DIAGONAL = False
```

Con la diagonal desactivada se fija `K(i,i)=1` y se ejecuta sólo el triángulo
superior. Para `m` filas se requieren:

```text
m(m-1)/2 circuitos
```

Ejemplos:

| Filas | Circuitos sin diagonal |
|---:|---:|
| 4 | 6 |
| 10 | 45 |
| 20 | 190 |

## Selección del backend

Sólo cambia `MATRIX_BACKEND`:

```python
MATRIX_BACKEND_OPTIONS = [
    "local_selene_statevector",
    "nexus_selene_statevector",
    "H1-1LE",
    "H1-Emulator",
    "H2-1LE",
    "H2-Emulator",
    "Helios-1E-lite",
]
```

### Selene local

```python
MATRIX_BACKEND = "local_selene_statevector"
```

No usa Nexus.

### Selene en Nexus

```python
MATRIX_BACKEND = "nexus_selene_statevector"
```

### H1 y H2

```python
MATRIX_BACKEND = "H1-1LE"
```

o:

```python
MATRIX_BACKEND = "H2-Emulator"
```

Estos backends no aceptan HUGR directamente. El notebook:

1. conserva el circuito como `pytket.Circuit`;
2. añade las mediciones;
3. lo sube a Nexus;
4. crea un compile job;
5. ejecuta el circuito compilado con `QuantinuumConfig`.

### Helios

```python
MATRIX_BACKEND = "Helios-1E-lite"
```

Helios usa el programa Guppy/HUGR mediante `HeliosConfig`.

## Consulta de jobs remotos

Después de ejecutar la celda `matrix-control`, no vuelvas a ejecutarla: eso
crearía recursos o jobs duplicados.

Ejecuta repetidamente la celda `matrix-poll-nexus`.

Para H1/H2 la celda:

1. consulta el compile job;
2. cuando termina, envía automáticamente el execute job;
3. consulta el execute job;
4. descarga resultados cuando aparece `COMPLETED`.

Para Selene/Helios consulta directamente el execute job.

Las consultas llaman al endpoint de estado de Nexus en cada ejecución; no
dependen del `last_status` almacenado en una referencia antigua.

## Archivos de resultados

Un solo par genera:

```text
data/runs/kernel_run_<id>.csv
```

Una matriz genera:

```text
data/runs/kernel_matrix_run_<id>.csv
```

Cada fila compacta incluye:

- backend y formato del programa;
- índices `row_i`, `row_j`;
- número de qubits;
- estado `00...0`;
- `zero_count`;
- `shots`;
- `kernel_rate`.

Los demás bitstrings no se guardan porque no participan directamente en el
kernel de fidelidad.

## Autenticación y proyectos Nexus

Al usar un backend remoto, `qnx.login()` solicitará autenticación. Cada persona
debe usar su propia cuenta.

El nombre del proyecto se configura con:

```python
MATRIX_PROJECT_NAME = "guppy-kernel-encoding"
```

Se puede cambiar antes de enviar jobs.

Nunca compartas:

- tokens;
- archivos de configuración con credenciales;
- cookies o sesiones;
- claves de API.

## Problemas frecuentes

### `ejecutar_local() got an unexpected keyword argument 'simulator'`

El kernel de Jupyter conservó una versión anterior del módulo.

```python
import importlib
import funciones_nexus

importlib.reload(funciones_nexus)
ejecutar_local = funciones_nexus.ejecutar_local
```

También puedes reiniciar el kernel y ejecutar desde la primera celda.

### `HUGR programs are only supported for Helios devices`

Se intentó enviar HUGR a H1/H2. En la versión incluida en este paquete,
H1/H2 se enrutan automáticamente mediante circuitos pytket. Recarga el
notebook desde disco y vuelve a ejecutar desde el inicio.

### El job aún no está `COMPLETED`

No vuelvas a enviar la celda de control. Reejecuta `matrix-poll-nexus`.

### Se muestran resultados antiguos

Recarga el notebook desde disco, reinicia el kernel y confirma que estás
trabajando dentro de esta carpeta.

### `K(i,j)=0`

Puede ser un resultado válido con fidelidad pequeña o una consecuencia de
pocos shots. Aumenta `MATRIX_SHOTS` para reducir la incertidumbre estadística.

## Recomendaciones antes de enviar

1. Empieza con dos o tres filas.
2. Usa pocos shots para validar el flujo.
3. Inspecciona el circuito y la cantidad planeada de ejecuciones.
4. Confirma el backend y la cuota.
5. Aumenta filas y shots gradualmente.
