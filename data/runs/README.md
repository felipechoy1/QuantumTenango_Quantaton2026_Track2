# Matrices kernel versionadas

Esta carpeta contiene la instantánea de matrices que usa el cuaderno. Los CSV
emplean `;` como separador, la primera columna es el índice y los valores son
estimaciones de fidelidad en `[0, 1]`.

| Prefijo | Feature map | Train | Test |
|---|---|---:|---:|
| `ZZ` | `zz` | 16 y 32 | 8 y 16 |
| `zyy` | `zyy` | 16 y 32 | 8 y 16 |
| `Custom` | `ry_cx_rx` | 16 y 32 | 8 y 16 |

La configuración reproducible declarada en el cuaderno es:

- backend local `local_selene_statevector`;
- 1,000 disparos por circuito;
- semilla 42;
- diagonal de `K_train` fijada en 1 sin ejecutar circuitos redundantes.

Las muestras son balanceadas y anidadas: train-16 está contenido en train-32,
y test-8 está contenido en test-16. Las matrices de distinto tamaño pueden
diferir ligeramente para un mismo par cuando provienen de estimaciones
independientes por disparos.

El 24 de julio de 2026 se validaron todas las matrices contra el circuito y el
orden de muestras actuales. Los archivos `ZZ_*_16.csv` se reconstruyeron como
los bloques anidados de los archivos `ZZ_*_32.csv`, que sí coincidían con el
circuito actual; esto eliminó una mezcla previa de ejecuciones sin cambiar la
definición del kernel.

Para comprobar formas, rangos, simetría, diagonal, espectro y coherencia entre
tamaños:

```bash
python validar_repositorio.py
```
