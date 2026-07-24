import hashlib
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import __version__ as sklearn_version
from sklearn.base import clone
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.svm import SVC
from tqdm import tqdm

import uuid
from pathlib import Path

import qnexus as qnx
from guppylang import guppy
from guppylang.std.builtins import array, comptime, result
from guppylang.std.quantum import measure_array, qubit
from pytket import Circuit
from pytket.passes import RemoveBarriers
from pytket.circuit.display import get_circuit_renderer
from IPython.display import HTML, display


def _fingerprint_datos(X, y):
    """Devuelve una huella estable del contenido, esquema y orden de X/y."""
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
    y_serie = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y))

    digest = hashlib.sha256()
    digest.update(b"qsvm-cache-v2")
    digest.update(repr(list(X_df.columns)).encode("utf-8"))
    digest.update(repr([str(dtype) for dtype in X_df.dtypes]).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(X_df, index=True).values.tobytes())
    digest.update(str(y_serie.name).encode("utf-8"))
    digest.update(str(y_serie.dtype).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(y_serie, index=True).values.tobytes())
    return digest.hexdigest()


def graficar_box_hist_grid(df, features, variables_por_fila=2, num_bins=10):
    """
    Grafica boxplot e histograma de cada variable en una sola figura,
    organizados en una cuadricula de N variables por fila.

    Cada variable ocupa dos subplots contiguos (boxplot e histograma).
    El eje Y del histograma esta en porcentaje (no en conteos), calculado
    sobre el total de registros de la variable (incluyendo los vacios),
    e incluye una barra adicional "Vacios" con el porcentaje de valores
    nulos, de forma que ambas partes sumen 100%.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame que contiene las columnas indicadas en `features`.
    features : list of str
        Nombres de las columnas de `df` a graficar.
    variables_por_fila : int, optional
        Cantidad de variables mostradas por fila de la cuadricula.
        Por defecto 2.
    num_bins : int, optional
        Numero de bins usados en cada histograma. Por defecto 10.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figura generada.
    axes : dict of str -> matplotlib.axes.Axes
        Diccionario de ejes retornado por `plt.subplot_mosaic`, con
        claves "box_{idx}" y "hist_{idx}" por cada variable (idx es la
        posicion de la variable en `features`).
    """
    mosaico = []
    for inicio in range(0, len(features), variables_por_fila):
        features_fila = features[inicio:inicio + variables_por_fila]
        fila = []
        for j in range(variables_por_fila):
            if j < len(features_fila):
                idx = inicio + j
                fila += [f"box_{idx}", f"hist_{idx}"]
            else:
                fila += [".", "."]
        mosaico.append(fila)

    fig, axes = plt.subplot_mosaic(
        mosaico,
        figsize=(18, 3.2 * len(mosaico)),
        constrained_layout=True
    )

    for idx, feature in enumerate(features):

        serie = df[feature]
        datos_validos = serie.dropna()
        cantidad_vacios = serie.isna().sum()
        total_registros = len(serie)
        porcentaje_vacios = cantidad_vacios / total_registros * 100

        ax_box = axes[f"box_{idx}"]
        ax_hist = axes[f"hist_{idx}"]

        # Boxplot
        ax_box.boxplot(
            datos_validos,
            vert=False
        )

        ax_box.set_title(
            f"{feature} - Boxplot"
        )

        ax_box.set_xlabel(feature)
        ax_box.set_yticks([])

        # Histograma (eje Y en porcentaje sobre el total de registros,
        # incluyendo los vacios, para que sumado con la barra "Vacios"
        # de 100%)
        pesos = np.full(len(datos_validos), 100 / total_registros)

        frecuencias, limites, _ = ax_hist.hist(
            datos_validos,
            bins=num_bins,
            weights=pesos,
            edgecolor="black",
            alpha=0.75
        )

        # Posicion de la barra de valores vacios
        ancho_bin = limites[1] - limites[0]
        posicion_vacios = limites[-1] + ancho_bin * 1.5

        ax_hist.bar(
            posicion_vacios,
            porcentaje_vacios,
            width=ancho_bin,
            color="gray",
            edgecolor="black"
        )

        # Etiqueta sobre la barra de vacios
        ax_hist.text(
            posicion_vacios,
            porcentaje_vacios,
            f"{porcentaje_vacios:.1f}%",
            ha="center",
            va="bottom"
        )

        # Conservamos ticks numericos y agregamos "Vacios"
        ticks_numericos = ax_hist.get_xticks()
        ticks_numericos = ticks_numericos[
            (ticks_numericos >= limites[0])
            & (ticks_numericos <= limites[-1])
        ]

        ax_hist.set_xticks([
            *ticks_numericos,
            posicion_vacios
        ])

        ax_hist.set_xticklabels([
            *[f"{valor:,.1f}" for valor in ticks_numericos],
            "Vacios"
        ], rotation=45)

        ax_hist.set_title(
            f"{feature} - Histograma"
        )

        ax_hist.set_xlabel(feature)
        ax_hist.set_ylabel("Porcentaje (%)")

    return fig, axes

def graficar_metricas_forward(resultados_forward, figsize=(10, 6)):
    """
    Grafica la evolucion de las metricas F1, AUC y Recall (validacion cruzada)
    en funcion del numero de variables seleccionadas por forward selection.

    Parameters
    ----------
    resultados_forward : pandas.DataFrame
        DataFrame con las columnas "numero_variables", "f1_cv", "auc_cv" y
        "recall_cv", tal como lo retorna la funcion `evaluar_forward`.
    figsize : tuple of int, optional
        Tamano de la figura en pulgadas (ancho, alto). Por defecto (10, 6).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figura generada.
    ax : matplotlib.axes.Axes
        Ejes de la grafica.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        resultados_forward["numero_variables"],
        resultados_forward["f1_cv"],
        color="firebrick",
        marker="o",
        linestyle="-",
        linewidth=1.5,
        markersize=5,
        label="F1-score"
    )

    ax.plot(
        resultados_forward["numero_variables"],
        resultados_forward["auc_cv"],
        color="steelblue",
        marker="s",
        linestyle="-",
        linewidth=1.5,
        markersize=5,
        label="AUC"
    )

    ax.plot(
        resultados_forward["numero_variables"],
        resultados_forward["recall_cv"],
        color="darkgoldenrod",
        marker="^",
        linestyle="-",
        linewidth=1.5,
        markersize=5,
        label="Recall"
    )

    ax.set_xlabel("Numero de variables seleccionadas")
    ax.set_ylabel("Metrica (validacion cruzada)")
    ax.set_title("Evolucion de metricas por numero de variables (Forward Selection)")
    ax.set_xticks(resultados_forward["numero_variables"])
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    return fig, ax

def evaluar_forward(
    X,
    y,
    modelo,
    cv,
    permitir_persistencia=True,
    forzar_reentrenamiento=False,
    directorio_cache="cache",
    param_grid=None,
    metrica_refit="f1",
    n_jobs=-1,
):
    """
    Evalua un modelo mediante forward selection incremental, calculando
    F1, AUC y Recall promedio por validacion cruzada para cada cantidad
    de variables seleccionadas.

    Para cada k en 1..n_variables, selecciona las k mejores variables con
    `SequentialFeatureSelector` (direction="forward", scoring="f1",
    usando `modelo` con sus hiperparametros fijos: cambiar hiperparametros
    dentro de la propia seleccion multiplicaria el costo por cada
    combinacion del grid). Cuando k es igual al total de variables, se
    usan todas sin ejecutar el selector.

    Una vez seleccionadas las variables de cada k, la evaluacion depende
    de `param_grid`:

    - `param_grid=None` (por defecto): se evalua `modelo` tal cual, con
      sus hiperparametros fijos, via `cross_validate`.
    - `param_grid` provisto: se corre `GridSearchCV` sobre esas variables
      para encontrar los mejores hiperparametros de *ese* k (refit por
      `metrica_refit`), y se reportan las metricas de CV en el mejor
      punto del grid, junto con los hiperparametros ganadores
      (columna "mejores_parametros"). Es decir, cada paso del forward
      encuentra sus propios mejores hiperparametros, en vez de evaluar
      todos los k con el mismo `modelo` fijo.

    Al ser un proceso costoso, el resultado se puede memorizar en disco:
    se arma una "line" (huella) con las columnas de X, su forma, el
    estimador (con sus hiperparametros), el esquema de validacion
    cruzada y el grid de hiperparametros (si aplica), y se hashea (md5)
    para nombrar el archivo de cache. Si ya existe un resultado con ese
    hash, se reutiliza en vez de reentrenar. El cache se guarda en JSON
    (no pickle), ya que el resultado es una tabla simple (numeros,
    listas de nombres de columnas y un diccionario de hiperparametros) y
    JSON no depende de la version de pandas/numpy que lo genero.

    Parameters
    ----------
    X : pandas.DataFrame
        Variables predictoras.
    y : pandas.Series or array-like
        Variable objetivo.
    modelo : sklearn estimator
        Estimador base a clonar y evaluar en cada paso (debe implementar
        la API de scikit-learn: fit/predict). Tambien es el estimador
        base de `GridSearchCV` cuando se pasa `param_grid`.
    cv : int, cross-validation generator or iterable
        Estrategia de validacion cruzada usada en el selector, en
        `cross_validate` y en `GridSearchCV`.
    permitir_persistencia : bool, optional
        Si es True (por defecto), el resultado se guarda en
        `directorio_cache` y, si ya existe un resultado para la misma
        combinacion de X, modelo, cv y param_grid, se reutiliza en lugar
        de recalcularlo.
    forzar_reentrenamiento : bool, optional
        Si es True, se borra el hash existente (si lo hay) y se vuelve a
        entrenar desde cero, sobrescribiendo el cache. Por defecto False.
    directorio_cache : str, optional
        Carpeta donde se guardan/leen los resultados persistidos. Por
        defecto "cache".
    param_grid : dict or None, optional
        Grid de hiperparametros para `GridSearchCV` (mismo formato que
        `sklearn.model_selection.GridSearchCV`), por ejemplo
        ``{"C": [0.1, 1, 10], "gamma": ["scale", "auto", 0.01]}``. Por
        defecto None: no se busca hiperparametros, se usa `modelo` tal
        cual (comportamiento identico al de antes de este parametro).
    metrica_refit : str, optional
        Metrica con la que `GridSearchCV` elige los mejores
        hiperparametros ("f1", "auc" o "recall"). Solo aplica si
        `param_grid` no es None. Por defecto "f1".

    Returns
    -------
    pandas.DataFrame
        Una fila por cada k evaluado, con las columnas:
        "numero_variables", "f1_cv", "auc_cv", "recall_cv" (promedios de
        validacion cruzada) y "variables" (lista de variables usadas).
        Si se paso `param_grid`, incluye ademas "mejores_parametros" (los
        hiperparametros ganadores de ese k).

    Raises
    ------
    ValueError
        Si `metrica_refit` no es "f1", "auc" ni "recall".
    """
    if metrica_refit not in {"f1", "auc", "recall"}:
        raise ValueError(
            f"metrica_refit debe ser 'f1', 'auc' o 'recall'; recibido '{metrica_refit}'."
        )

    ruta_cache = None

    if permitir_persistencia:
        line = "|".join([
            _fingerprint_datos(X, y),
            str(X.columns.tolist()),
            repr(modelo.get_params()),
            repr(cv),
            repr(param_grid),
            metrica_refit if param_grid is not None else "",
            str(n_jobs),
            sklearn_version,
        ])

        hash_line = hashlib.sha256(line.encode("utf-8")).hexdigest()

        os.makedirs(directorio_cache, exist_ok=True)
        ruta_cache = os.path.join(
            directorio_cache,
            f"evaluar_forward_{hash_line}.json"
        )

        if forzar_reentrenamiento and os.path.exists(ruta_cache):
            print(f"[evaluar_forward] Cache existente en '{ruta_cache}': se reescribira (forzar_reentrenamiento=True).")
            os.remove(ruta_cache)

        elif not forzar_reentrenamiento and os.path.exists(ruta_cache):
            print(f"[evaluar_forward] Cache existente en '{ruta_cache}': se cargan los resultados guardados.")
            return pd.read_json(ruta_cache, orient="records")

        else:
            print(f"[evaluar_forward] Sin cache previo en '{ruta_cache}': se entrena desde cero.")

    resultados = []
    total_variables = X.shape[1]

    for k in tqdm(range(1, total_variables + 1), desc="evaluar_forward"):

        # Forward selection (con los hiperparametros fijos de `modelo`)
        if k < total_variables:

            selector = SequentialFeatureSelector(
                estimator=clone(modelo),
                n_features_to_select=k,
                direction="forward",
                scoring="f1",
                cv=cv,
                n_jobs=n_jobs
            )

            selector.fit(X, y)

            variables = X.columns[
                selector.get_support()
            ].tolist()

        else:
            variables = X.columns.tolist()

        fila = {"numero_variables": k, "variables": variables}

        if param_grid is None:
            # F1, AUC y recall promedio mediante CV, con modelo tal cual.
            metricas_cv = cross_validate(
                estimator=clone(modelo),
                X=X[variables],
                y=y,
                cv=cv,
                scoring={
                    "f1": "f1",
                    "auc": "roc_auc",
                    "recall":"recall"
                },
                n_jobs=n_jobs
            )
            fila["f1_cv"] = metricas_cv["test_f1"].mean()
            fila["auc_cv"] = metricas_cv["test_auc"].mean()
            fila["recall_cv"] = metricas_cv["test_recall"].mean()

        else:
            # Mejores hiperparametros para este k especifico.
            grid = GridSearchCV(
                estimator=clone(modelo),
                param_grid=param_grid,
                scoring={
                    "f1": "f1",
                    "auc": "roc_auc",
                    "recall": "recall"
                },
                refit=metrica_refit,
                cv=cv,
                n_jobs=n_jobs
            )
            grid.fit(X[variables], y)

            mejor = grid.cv_results_
            idx = grid.best_index_
            fila["f1_cv"] = mejor["mean_test_f1"][idx]
            fila["auc_cv"] = mejor["mean_test_auc"][idx]
            fila["recall_cv"] = mejor["mean_test_recall"][idx]
            fila["mejores_parametros"] = grid.best_params_

        resultados.append(fila)

    resultado_df = pd.DataFrame(resultados)

    if permitir_persistencia:
        resultado_df.to_json(ruta_cache, orient="records", indent=2)

    return resultado_df


def graficar_matrices_confusion(
    y_train,
    y_pred_train,
    y_test,
    y_pred_test,
    display_labels=("No potable", "Potable"),
    figsize=(10, 4.5)
):
    """
    Grafica las matrices de confusion de train y test lado a lado en una
    sola figura, con estilo formal (titulos y etiquetas consistentes)
    apto para un paper.

    Cada matriz se colorea por conteo (no por porcentaje) y con su
    propia escala de color independiente (cada una con su colorbar), de
    forma que dentro de cada matriz se distinga claramente que cuadrante
    concentra mas poblacion, sin forzar a train y test a compartir
    escala.

    Parameters
    ----------
    y_train : array-like
        Etiquetas reales del conjunto de entrenamiento.
    y_pred_train : array-like
        Etiquetas predichas para el conjunto de entrenamiento.
    y_test : array-like
        Etiquetas reales del conjunto de prueba.
    y_pred_test : array-like
        Etiquetas predichas para el conjunto de prueba.
    display_labels : tuple of str, optional
        Nombres de las clases, en el orden correspondiente a las
        etiquetas. Por defecto ("No potable", "Potable").
    figsize : tuple of int, optional
        Tamano de la figura en pulgadas (ancho, alto). Por defecto
        (10, 4.5).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figura generada.
    axes : numpy.ndarray of matplotlib.axes.Axes
        Los dos ejes (train, test) retornados por `plt.subplots`.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    disp_train = ConfusionMatrixDisplay.from_predictions(
        y_train,
        y_pred_train,
        display_labels=display_labels,
        cmap="Blues",
        ax=axes[0],
        colorbar=True,
        text_kw={"fontsize": 11}
    )
    axes[0].set_title("Train", fontsize=12, fontweight="bold")

    disp_test = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred_test,
        display_labels=display_labels,
        cmap="Blues",
        ax=axes[1],
        colorbar=True,
        text_kw={"fontsize": 11}
    )
    axes[1].set_title("Test", fontsize=12, fontweight="bold")

    for ax in axes:
        ax.set_xlabel("Prediccion", fontsize=10)
        ax.set_ylabel("Real", fontsize=10)
        ax.grid(False)

    fig.suptitle("Matrices de confusion", fontsize=14, fontweight="bold")

    return fig, axes


def classification_metrics(y_true, y_pred, scores=None):
    """
    Calcula un conjunto estandar de metricas de clasificacion binaria.

    Parameters
    ----------
    y_true : array-like
        Etiquetas reales.
    y_pred : array-like
        Etiquetas predichas por el modelo.
    scores : array-like, optional
        Puntajes o probabilidades usados para calcular el AUC (por
        ejemplo, la salida de `decision_function` o `predict_proba`).
        Si es None, "roc_auc" se retorna como NaN.

    Returns
    -------
    dict
        Diccionario con las claves "accuracy", "precision", "recall",
        "f1" y "roc_auc".
    """
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    result["roc_auc"] = (
        roc_auc_score(y_true, scores) if scores is not None else np.nan
    )
    return result


def comparar_metricas_train_test(metrics_train, metrics_test):
    """
    Construye una tabla comparativa de metricas de train vs test,
    incluyendo la brecha absoluta y relativa (indicador de overfitting),
    lista para mostrarse en el notebook.

    Parameters
    ----------
    metrics_train : dict
        Metricas del conjunto de entrenamiento, tal como las retorna
        `classification_metrics`.
    metrics_test : dict
        Metricas del conjunto de prueba, tal como las retorna
        `classification_metrics`.

    Returns
    -------
    pandas.io.formats.style.Styler
        Tabla con columnas "train", "test", "brecha_overfit" (train -
        test) y "brecha_relativa_pct" (brecha_overfit / train, en
        porcentaje), formateada para su despliegue directo.
    """
    metrics_comparison = pd.DataFrame({
        "train": metrics_train,
        "test": metrics_test
    })

    # Diferencia absoluta
    metrics_comparison["brecha_overfit"] = (
        metrics_comparison["train"]
        - metrics_comparison["test"]
    )

    # Diferencia relativa respecto a train
    metrics_comparison["brecha_relativa_pct"] = np.where(
        metrics_comparison["train"] != 0,
        (
            metrics_comparison["brecha_overfit"]
            / metrics_comparison["train"]
        ) * 100,
        np.nan
    )

    metrics_comparison.index.name = "metrica"

    return metrics_comparison.style.format({
        "train": "{:.3f}",
        "test": "{:.3f}",
        "brecha_overfit": "{:.3f}",
        "brecha_relativa_pct": "{:.2f}%"
    })


def guardar_datasets_excel(
    df,
    X_train,
    X_test,
    y_train,
    y_test,
    scaler,
    features,
    objetivo,
    ruta_excel,
    columnas_imputables=("ph", "Sulfate", "Trihalomethanes")
):
    """
    Guarda en un unico archivo Excel tres versiones del dataset
    (originales, imputados, normalizados), cada una en su propia hoja,
    con train y test unificados (X + Y) y columnas de control para
    distinguir particion e imputacion.

    "originales" se reconstruye indexando `df` (datos crudos, con
    nulos) por los indices de `X_train`/`X_test`. "imputados" se
    recupera invirtiendo `scaler` sobre `X_train`/`X_test` (recupera el
    estado post-imputacion/pre-escalado). "normalizados" es el estado
    actual (ya imputado y escalado) de `X_train`/`X_test`.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset original completo (con nulos), tal como se leyo antes
        de cualquier imputacion o split.
    X_train : pandas.DataFrame
        Variables predictoras de entrenamiento, en su estado actual
        (imputado y escalado).
    X_test : pandas.DataFrame
        Variables predictoras de prueba, en su estado actual (imputado
        y escalado).
    y_train : pandas.Series
        Variable objetivo de entrenamiento.
    y_test : pandas.Series
        Variable objetivo de prueba.
    scaler : sklearn transformer
        Escalador (por ejemplo `StandardScaler`) ya ajustado sobre
        `X_train`, usado para invertir el escalado y recuperar la
        version "imputados".
    features : list of str
        Nombres de las columnas originales (crudas) a usar para la hoja
        "originales".
    objetivo : str
        Nombre de la columna objetivo, agregada a cada tabla.
    ruta_excel : str
        Ruta del archivo `.xlsx` de salida. La carpeta contenedora se
        crea si no existe.
    columnas_imputables : tuple of str, optional
        Columnas sobre las que se aplico imputacion, usadas para
        calcular "_Imputation_" y "_ImputationCol_". Por defecto
        ("ph", "Sulfate", "Trihalomethanes").

    Returns
    -------
    str
        La ruta del archivo Excel generado.
    """
    def agregar_indicadores(X_part, y_part, part_ind):
        tabla = X_part.copy()
        tabla[objetivo] = y_part.values
        tabla["_PartInd_"] = part_ind

        nulos = df.loc[X_part.index, list(columnas_imputables)].isna()
        tabla["_Imputation_"] = nulos.any(axis=1).astype(int)
        tabla["_ImputationCol_"] = nulos.apply(
            lambda fila: ",".join(fila.index[fila]), axis=1
        )
        return tabla

    # Originales: valores crudos (con nulos), tal cual antes de imputar
    tabla_originales = pd.concat([
        agregar_indicadores(df.loc[X_train.index, features], y_train, 0),
        agregar_indicadores(df.loc[X_test.index, features], y_test, 1),
    ], ignore_index=True)

    # Imputados: se recupera invirtiendo el escalado ya ajustado
    X_train_imputado = pd.DataFrame(
        scaler.inverse_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_imputado = pd.DataFrame(
        scaler.inverse_transform(X_test), columns=X_test.columns, index=X_test.index
    )

    tabla_imputados = pd.concat([
        agregar_indicadores(X_train_imputado, y_train, 0),
        agregar_indicadores(X_test_imputado, y_test, 1),
    ], ignore_index=True)

    # Normalizados: estado actual (ya imputado y escalado)
    tabla_normalizados = pd.concat([
        agregar_indicadores(X_train, y_train, 0),
        agregar_indicadores(X_test, y_test, 1),
    ], ignore_index=True)

    directorio = os.path.dirname(ruta_excel)
    if directorio:
        os.makedirs(directorio, exist_ok=True)

    with pd.ExcelWriter(ruta_excel, engine="openpyxl") as writer:
        tabla_originales.to_excel(writer, sheet_name="originales", index=False)
        tabla_imputados.to_excel(writer, sheet_name="imputados", index=False)
        tabla_normalizados.to_excel(writer, sheet_name="normalizados", index=False)

    return ruta_excel


def guardar_muestreo_excel(tabla_muestreo, ruta_excel, sheet_name="muestreos"):
    """
    Agrega (o reemplaza) la hoja `sheet_name` en un archivo Excel ya
    existente, con la tabla de muestreo jerarquico balanceado.

    Requiere que `ruta_excel` ya exista (por ejemplo, generado antes por
    `guardar_datasets_excel`); esta funcion solo agrega/reemplaza una hoja,
    no crea el archivo desde cero.

    Parameters
    ----------
    tabla_muestreo : pandas.DataFrame
        Tabla con la columna "_Muestreo_" (por ejemplo, el resultado de
        `asignar_muestreo_balanceado`).
    ruta_excel : str
        Ruta del archivo `.xlsx` ya existente.
    sheet_name : str, optional
        Nombre de la hoja a agregar/reemplazar. Por defecto "muestreos".

    Returns
    -------
    str
        La ruta del archivo Excel.
    """
    with pd.ExcelWriter(
        ruta_excel, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        tabla_muestreo.to_excel(writer, sheet_name=sheet_name, index=False)

    return ruta_excel


def asignar_muestreo_balanceado(
    df,
    tamanos_por_particion,
    objetivo="Potability",
    particion="_PartInd_",
    columna="_Muestreo_",
    columna_imputacion="_Imputation_",
    random_state=42,
):
    """
    Agrega una columna de muestreo jerarquico balanceado por particion.

    Para cada valor de `particion` se dibuja una secuencia de muestras
    **anidadas** y **balanceadas** (mismo numero de filas por clase de
    `objetivo`, es decir 50/50 en un objetivo binario; balanceado, no
    estratificado). La muestra mas grande marca `columna` = 1; de ella se
    extrae una submuestra que sobrescribe la marca a 2; de esa, otra que la
    sobrescribe a 3; y asi para cada tamano. Las filas de la particion que
    no entran en la muestra mas grande quedan en 0.

    Como cada nivel sobrescribe al anterior, el valor final de `columna`
    indica el nivel mas profundo que alcanzo cada fila. Esto permite
    reconstruir cada muestra: las filas de la muestra de nivel k son las que
    tienen `columna` >= k (por el anidamiento).

    Regla de datos virgenes: el muestreo se hace unicamente sobre filas
    **sin imputacion** (`columna_imputacion` == 0). Al trabajar con pocas
    filas conviene centrarse en datos que no fueron tocados por la
    imputacion. Basta filtrar el pool del nivel 1: los niveles siguientes
    son submuestras de el, asi que heredan la regla.

    Parameters
    ----------
    df : pandas.DataFrame
        Datos con las columnas `objetivo`, `particion` y
        `columna_imputacion`.
    tamanos_por_particion : dict
        Mapea cada valor de `particion` a una lista de tamanos totales de
        muestra, de mayor a menor y anidados (por ejemplo
        ``{0: [64, 32, 16], 1: [32, 16, 8]}``). Cada tamano debe ser
        divisible entre el numero de clases y estrictamente menor que el
        anterior.
    objetivo : str, optional
        Columna objetivo sobre la que se balancea. Por defecto "Potability".
    particion : str, optional
        Columna que define las particiones. Por defecto "_PartInd_".
    columna : str, optional
        Nombre de la columna de muestreo a crear. Por defecto "_Muestreo_".
    columna_imputacion : str or None, optional
        Columna que marca filas con algun valor imputado (1) o virgenes
        (0). Por defecto "_Imputation_". Con None se desactiva la regla y
        se muestrea sobre toda la particion.
    random_state : int, optional
        Semilla para reproducibilidad. Por defecto 42.

    Returns
    -------
    pandas.DataFrame
        Copia de `df` con la columna `columna` agregada (enteros; 0 para las
        filas fuera de la muestra).

    Raises
    ------
    ValueError
        Si falta `columna_imputacion` en `df`, algun tamano no es divisible
        entre el numero de clases, los tamanos no van en orden
        estrictamente decreciente, o no hay suficientes filas de alguna
        clase para una muestra balanceada.
    """
    resultado = df.copy()
    resultado[columna] = 0
    rng = np.random.RandomState(random_state)

    columnas_requeridas = {objetivo, particion}
    faltantes = columnas_requeridas.difference(resultado.columns)
    if faltantes:
        raise ValueError(
            f"Faltan columnas requeridas en df: {sorted(faltantes)}."
        )

    if columna_imputacion is not None and columna_imputacion not in resultado.columns:
        raise ValueError(
            f"La columna '{columna_imputacion}' no existe en df; se necesita para "
            f"muestrear solo filas sin imputacion (o pasa columna_imputacion=None "
            f"para desactivar la regla)."
        )

    for valor_particion, tamanos in tamanos_por_particion.items():
        tamanos = list(tamanos)
        filas_particion = resultado.loc[resultado[particion] == valor_particion]
        if columna_imputacion is not None:
            filas_particion = filas_particion.loc[
                filas_particion[columna_imputacion] == 0
            ]
        clases = sorted(filas_particion[objetivo].unique())
        n_clases = len(clases)
        if n_clases < 2:
            raise ValueError(
                f"Particion {valor_particion}: se necesitan al menos dos clases "
                f"para construir una muestra balanceada; encontradas {clases}."
            )

        for tamano in tamanos:
            if tamano % n_clases != 0:
                raise ValueError(
                    f"Particion {valor_particion}: el tamano {tamano} no es divisible "
                    f"entre {n_clases} clases; no se puede balancear."
                )
        for anterior, siguiente in zip(tamanos, tamanos[1:]):
            if siguiente >= anterior:
                raise ValueError(
                    f"Particion {valor_particion}: los tamanos deben ir en orden "
                    f"estrictamente decreciente (muestras anidadas); recibido {tamanos}."
                )

        idx_nivel_previo = None
        for nivel, tamano in enumerate(tamanos, start=1):
            por_clase = tamano // n_clases
            pool = filas_particion if nivel == 1 else resultado.loc[idx_nivel_previo]

            partes = []
            for clase in clases:
                grupo = pool[pool[objetivo] == clase]
                if len(grupo) < por_clase:
                    raise ValueError(
                        f"Particion {valor_particion}, clase {clase}: hay {len(grupo)} "
                        f"filas disponibles en el nivel {nivel}, se necesitan {por_clase} "
                        f"para una muestra balanceada de {tamano}."
                    )
                partes.append(grupo.sample(n=por_clase, random_state=rng))

            idx_nivel = pd.concat(partes).index
            resultado.loc[idx_nivel, columna] = nivel
            idx_nivel_previo = idx_nivel

    return resultado


def zz_feature_map(x):
    """
    Construye el feature map ZZ ``U(x)`` enteramente en pytket.

    Aplica una capa de Hadamard, rotaciones Rz individuales proporcionales
    a cada feature y correlaciones ZZ (CX-Rz-CX) entre cada par de qubits.
    pytket expresa los angulos en medias vueltas, por eso cada angulo se
    divide entre pi.

    Parameters
    ----------
    x : array-like
        Vector de features escaladas de una observacion; cada feature se
        codifica en un qubit.

    Returns
    -------
    pytket.Circuit
        Circuito U(x) con barreras entre bloques (utiles para inspeccion
        visual; se eliminan antes de ejecutar).
    """
    x = np.asarray(x, dtype=float)
    n_qubits = len(x)

    circuit = Circuit(n_qubits, name="ZZ Feature Map")
    qubits = list(range(n_qubits))

    # Embedding individual
    for i in range(n_qubits):
        circuit.H(i)

    # pytket expresa los angulos en medias vueltas.
    for i in range(n_qubits):
        circuit.Rz(2 * x[i] / np.pi, i)

    circuit.add_barrier(qubits)

    # Correlaciones ZZ
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            angle = 2 * (np.pi - x[i]) * (np.pi - x[j])
            circuit.CX(i, j)
            circuit.Rz(angle / np.pi, j)
            circuit.CX(i, j)
            circuit.add_barrier(qubits)

    return circuit


def agregar_exp_yy(circuit, left, right, parameter):
    """
    Aplica ``exp(-i * phi * Y_left Y_right)`` con compuertas elementales.

    Cambia de la base Y a la base Z (``Sdg`` + ``H``), aplica la fase
    conjunta como una interaccion ZZ por paridad (``CX-Rz-CX``) y regresa
    a la base Y (``H`` + ``S``). No usa PauliExpBox, por lo que el circuito
    queda inspeccionable puerta por puerta.

    Parameters
    ----------
    circuit : pytket.Circuit
        Circuito sobre el que se agregan las compuertas (in place).
    left, right : int
        Qubits sobre los que actua la interaccion YY.
    parameter : float
        Angulo de la rotacion Rz central, ya expresado en medias vueltas
        (``2 * phi / pi``).
    """
    # Cambio de base Y -> Z
    circuit.Sdg(left)
    circuit.Sdg(right)
    circuit.H(left)
    circuit.H(right)

    # Exponencial ZZ mediante paridad
    circuit.CX(left, right)
    circuit.Rz(parameter, right)
    circuit.CX(left, right)

    # Regreso de base Z -> Y
    circuit.H(left)
    circuit.H(right)
    circuit.S(left)
    circuit.S(right)


def pauli_feature_map_zyy(x):
    """
    Construye el feature map Pauli ``Z + YY`` (entrelazamiento lineal).

    Aplica una capa de Hadamard, rotaciones Rz individuales (terminos Z,
    ``phi_i(x) = x_i``) y correlaciones YY entre qubits vecinos
    (``phi_ij(x) = (pi - x_i)(pi - x_j)``) expandidas con compuertas
    elementales via `agregar_exp_yy` (sin PauliExpBox). pytket expresa los
    angulos en medias vueltas, por eso cada angulo se divide entre pi.

    Parameters
    ----------
    x : array-like
        Vector de features escaladas de una observacion; cada feature se
        codifica en un qubit.

    Returns
    -------
    pytket.Circuit
        Circuito U(x) con barreras entre bloques (utiles para inspeccion
        visual; se eliminan antes de ejecutar).
    """
    x = np.asarray(x, dtype=float)
    n_qubits = len(x)

    circuit = Circuit(n_qubits, name="Pauli Z+YY lineal")
    qubits = list(range(n_qubits))

    # Superposicion inicial
    for i in range(n_qubits):
        circuit.H(i)

    # Terminos individuales Z; pytket usa medias vueltas.
    for i in range(n_qubits):
        circuit.Rz(2 * x[i] / np.pi, i)

    circuit.add_barrier(qubits)

    # Interacciones YY entre qubits vecinos (entrelazamiento lineal)
    for left in range(n_qubits - 1):
        right = left + 1
        phi = (np.pi - x[left]) * (np.pi - x[right])
        agregar_exp_yy(circuit, left, right, 2 * phi / np.pi)
        circuit.add_barrier(qubits)

    return circuit


def feature_map_ry_cx_rx(x):
    """
    Construye el feature map personalizado ``Ry -> cadena CX -> Rx``.

    Cada qubit recibe ``Ry(x_i)``, luego se aplica la cadena lineal de
    ``CX`` (0->1, 1->2, ...) y finalmente cada qubit recibe ``Rx(x_i)``.
    pytket expresa las rotaciones en medias vueltas, por eso el angulo
    fisico ``x_i`` se escribe como ``x_i / pi``.

    Parameters
    ----------
    x : array-like
        Vector de features escaladas de una observacion; cada feature se
        codifica en un qubit.

    Returns
    -------
    pytket.Circuit
        Circuito U(x) con barreras entre bloques (utiles para inspeccion
        visual; se eliminan antes de ejecutar).
    """
    x = np.asarray(x, dtype=float)
    n_qubits = len(x)

    circuit = Circuit(n_qubits, name="RY-CX-RX lineal")
    qubits = list(range(n_qubits))

    # Rotaciones Ry individuales
    for i in range(n_qubits):
        circuit.Ry(x[i] / np.pi, i)

    circuit.add_barrier(qubits)

    # Cadena lineal de CX
    for control in range(n_qubits - 1):
        circuit.CX(control, control + 1)

    circuit.add_barrier(qubits)

    # Rotaciones Rx individuales
    for i in range(n_qubits):
        circuit.Rx(x[i] / np.pi, i)

    return circuit


# Registro de feature maps disponibles para el kernel. La clave es el
# identificador que se pasa como parametro `feature_map`; el valor es el
# constructor U(x) en pytket. Todos comparten la misma tuberia (mismo
# `kernel_circuit`, mismo programa guppy, misma extraccion de P(00...0)).
FEATURE_MAPS = {
    "zz": zz_feature_map,
    "zyy": pauli_feature_map_zyy,
    "ry_cx_rx": feature_map_ry_cx_rx,
}


def obtener_feature_map(feature_map):
    """
    Resuelve el parametro `feature_map` a su constructor U(x).

    Parameters
    ----------
    feature_map : str or callable
        Clave de `FEATURE_MAPS` ("zz", "zyy" o "ry_cx_rx") o directamente
        una funcion ``f(x) -> pytket.Circuit``.

    Returns
    -------
    callable
        La funcion constructora del feature map.

    Raises
    ------
    ValueError
        Si `feature_map` es una cadena que no esta en `FEATURE_MAPS`.
    """
    if callable(feature_map):
        return feature_map
    try:
        return FEATURE_MAPS[feature_map]
    except KeyError:
        opciones = ", ".join(sorted(FEATURE_MAPS))
        raise ValueError(
            f"feature_map '{feature_map}' no reconocido. "
            f"Opciones: {opciones}."
        )


def parsear_familia(familia):
    """Descompone una entrada de FAMILIAS en (feature_map, tamano, ruta_train, ruta_test).

    Cada entrada es (feature_map, tamano) o
    (feature_map, tamano, ruta_train, ruta_test). ruta_train/ruta_test son
    las rutas (str) a los csv con las matrices K_train/K_test ya
    calculadas en una ejecucion anterior; usa "" para la que todavia no
    exista. Formatos validos: 2 elementos, o 4 elementos con el 3ro y/o
    4to en blanco ("").

    - Si ambas rutas estan presentes, la familia ya esta completa: no se
      calcula nada (idempotencia).
    - Si falta una, solo esa matriz se debe calcular; la otra se reutiliza
      desde su ruta.
    - Si no hay 3er/4to elemento, ambas quedan pendientes de calcular.

    Devuelve None (no "") para la ruta que falte, para simplificar los
    chequeos posteriores (``if ruta_train:``).

    Parameters
    ----------
    familia : tuple
        Entrada de FAMILIAS, con 2 o 4 elementos.

    Returns
    -------
    tuple
        (feature_map, tamano, ruta_train, ruta_test).

    Raises
    ------
    ValueError
        Si `familia` no tiene 2 ni 4 elementos, o si ruta_train/ruta_test
        no son strings.
    """
    if len(familia) == 2:
        feature_map, tamano = familia
        return feature_map, tamano, None, None
    if len(familia) == 4:
        feature_map, tamano, ruta_train, ruta_test = familia
        for nombre_campo, ruta in (("ruta_train", ruta_train), ("ruta_test", ruta_test)):
            if not isinstance(ruta, str):
                raise ValueError(
                    f'{nombre_campo} de {familia!r} debe ser un string (usa "" si no '
                    f'aplica); recibido {ruta!r}.'
                )
        return feature_map, tamano, (ruta_train or None), (ruta_test or None)
    raise ValueError(
        f"Cada entrada de FAMILIAS debe tener 2 o 4 elementos "
        f"(feature_map, tamano) o (feature_map, tamano, ruta_train, ruta_test); "
        f"recibido {familia!r} con {len(familia)} elementos."
    )


def nivel_muestreo(tamano, nivel_por_tamano):
    """Resuelve el nivel de `_Muestreo_` correspondiente a un tamano de train.

    Parameters
    ----------
    tamano : int
        Tamano de train de la familia (por ejemplo 16 o 32).
    nivel_por_tamano : dict
        Mapeo tamano -> nivel de muestreo (columna `_Muestreo_`).

    Returns
    -------
    int
        Nivel de muestreo correspondiente.

    Raises
    ------
    ValueError
        Si `tamano` no esta en `nivel_por_tamano`.
    """
    if tamano not in nivel_por_tamano:
        raise ValueError(
            f"Tamano {tamano} sin nivel de muestreo definido; agrega la entrada "
            f"correspondiente en nivel_por_tamano."
        )
    return nivel_por_tamano[tamano]


def contar_registros_familia(tamano, nivel_por_tamano, muestreo_df):
    """Cuenta filas de train/test disponibles para el nivel de muestreo de `tamano`.

    Parameters
    ----------
    tamano : int
        Tamano de train de la familia.
    nivel_por_tamano : dict
        Mapeo tamano -> nivel de muestreo, usado por `nivel_muestreo`.
    muestreo_df : pandas.DataFrame
        Hoja `muestreos` de `dataset_v1.xlsx`, con las columnas
        `_PartInd_` y `_Muestreo_`.

    Returns
    -------
    tuple of int
        (n_train, n_test).
    """
    nivel = nivel_muestreo(tamano, nivel_por_tamano)
    n_train = ((muestreo_df["_PartInd_"] == 0) & (muestreo_df["_Muestreo_"] >= nivel)).sum()
    n_test = ((muestreo_df["_PartInd_"] == 1) & (muestreo_df["_Muestreo_"] >= nivel)).sum()
    return int(n_train), int(n_test)


def construir_circuitos_ejemplo(x_ejemplo, n_qubits):
    """Construye U(x) y U(x) dagger de ejemplo para cada feature map registrado.

    Un circuito de ejemplo por feature map basta para inspeccion visual:
    el tamano de muestra (16 vs 32) no cambia el circuito, solo cuantos
    pares se corren despues.

    Parameters
    ----------
    x_ejemplo : array-like
        Vector de features de una observacion (misma fila para los 3
        feature maps).
    n_qubits : int
        Numero de qubits esperado (variables seleccionadas en el Paso 1).

    Returns
    -------
    tuple of dict
        (circuitos_u, circuitos_u_dagger), cada uno mapeando el nombre del
        feature map a su `pytket.Circuit`.

    Raises
    ------
    AssertionError
        Si algun feature map no genera un circuito con `n_qubits` qubits.
    """
    circuitos_u = {}
    circuitos_u_dagger = {}
    for nombre_fm, constructor in FEATURE_MAPS.items():
        circuito_u = constructor(x_ejemplo)
        assert circuito_u.n_qubits == n_qubits, (
            f"El feature map '{nombre_fm}' no coincide con n_qubits={n_qubits}."
        )
        circuito_u.name = f"{nombre_fm} - U(x)"

        circuito_u_dagger = circuito_u.dagger()
        circuito_u_dagger.name = f"{nombre_fm} - U(x) dagger"

        circuitos_u[nombre_fm] = circuito_u
        circuitos_u_dagger[nombre_fm] = circuito_u_dagger
    return circuitos_u, circuitos_u_dagger


def mostrar_inspeccion_feature_map(
    circuitos_u,
    circuitos_u_dagger,
    alto_cabecera=150,
    alto_por_qubit=72,
):
    """Renderiza los circuitos U(x)/U(x) dagger de cada familia, apilados.

    Muestra los 6 circuitos (U y U dagger de cada feature map) uno debajo
    del otro, cada uno precedido de un titulo que lo identifica. Cada
    circuito se renderiza por separado (una llamada al renderer por
    circuito), no como una lista con ``orient="column"``: el render de
    pytket mete la lista en un solo iframe de altura fija y los circuitos
    se recortan y traslapan. Renderizando de a uno, cada circuito recibe
    su propia ventana navegable (zoom/scroll propios del render) con la
    altura ajustada a su numero de qubits.

    Parameters
    ----------
    circuitos_u : dict
        Nombre de feature map -> `pytket.Circuit` de U(x), tal como los
        devuelve `construir_circuitos_ejemplo`.
    circuitos_u_dagger : dict
        Nombre de feature map -> `pytket.Circuit` de U(x) dagger.
    alto_cabecera : int, optional
        Pixeles reservados para la barra de controles del render (parte
        que no depende del numero de qubits). Por defecto 150.
    alto_por_qubit : int, optional
        Pixeles que ocupa cada fila de qubit en el render. Por defecto 72.
        Junto con `alto_cabecera` fija la altura de la ventana de cada
        circuito para que se vea completo sin recortes ni exceso de
        espacio en blanco.
    """
    renderer = get_circuit_renderer()

    def _render_circuito(titulo, circuito):
        display(HTML(
            "<div style='font-weight:600;font-size:14px;margin:16px 0 4px;"
            f"font-family:sans-serif'>{titulo}</div>"
        ))
        renderer.config.min_height = f"{alto_cabecera + circuito.n_qubits * alto_por_qubit}px"
        renderer.render_circuit_jupyter(circuito)

    for nombre_fm in circuitos_u:
        _render_circuito(f"{nombre_fm} · U(x)", circuitos_u[nombre_fm])
        _render_circuito(f"{nombre_fm} · U(x)†", circuitos_u_dagger[nombre_fm])


def evaluar_grid_search(
    X,
    y,
    modelo,
    param_grid,
    cv,
    scoring,
    refit,
    permitir_persistencia=True,
    forzar_reentrenamiento=False,
    directorio_cache="cache",
    n_jobs=-1,
):
    """Ejecuta (o recupera de cache) un GridSearchCV sobre variables fijas.

    Igual que `evaluar_forward`, la busqueda es costosa pero siempre da el
    mismo resultado para los mismos datos/modelo/grid, asi que se memoriza
    en disco: se arma una huella con las columnas de X, su forma, el
    estimador base (con sus hiperparametros), param_grid, cv, scoring y
    refit, y se hashea (md5) para nombrar el archivo de cache. Si ya existe
    un resultado con ese hash, se reutiliza en vez de repetir la busqueda.

    El cache solo guarda `cv_results_` y los mejores hiperparametros (JSON,
    no pickle, por la misma razon que en `evaluar_forward`); el estimador
    final SI se reentrena siempre con esos hiperparametros ganadores (un
    solo fit, barato) porque un modelo de sklearn no se guarda en JSON.

    Parameters
    ----------
    X : pandas.DataFrame
        Variables predictoras (ya seleccionadas).
    y : pandas.Series or array-like
        Variable objetivo.
    modelo : sklearn estimator
        Estimador base a clonar (sus hiperparametros fijos son el punto de
        partida; `param_grid` sobreescribe los que este buscando).
    param_grid : dict
        Grid de hiperparametros de `GridSearchCV`.
    cv : int, cross-validation generator or iterable
        Estrategia de validacion cruzada.
    scoring : dict
        Metricas de `GridSearchCV` (formato `scikit-learn`).
    refit : str
        Metrica de `scoring` usada para elegir los mejores hiperparametros.
    permitir_persistencia : bool, optional
        Si es True (por defecto), el resultado se guarda en
        `directorio_cache` y se reutiliza si ya existe uno para la misma
        combinacion de X, modelo, param_grid, cv, scoring y refit.
    forzar_reentrenamiento : bool, optional
        Si es True, se borra el hash existente (si lo hay) y se repite la
        busqueda desde cero, sobrescribiendo el cache. Por defecto False.
    directorio_cache : str, optional
        Carpeta donde se guardan/leen los resultados persistidos. Por
        defecto "cache".

    Returns
    -------
    tuple
        (cv_results_df, mejor_estimador, mejores_parametros):
        `cv_results_df` es `pd.DataFrame(grid.cv_results_)` completo,
        `mejor_estimador` ya esta reentrenado con `mejores_parametros`
        sobre todo `X`/`y`.
    """
    ruta_cache = None

    if permitir_persistencia:
        line = "|".join([
            _fingerprint_datos(X, y),
            str(X.columns.tolist()),
            repr(modelo.get_params()),
            repr(param_grid),
            repr(cv),
            repr(scoring),
            refit,
            str(n_jobs),
            sklearn_version,
        ])

        hash_line = hashlib.sha256(line.encode("utf-8")).hexdigest()

        os.makedirs(directorio_cache, exist_ok=True)
        ruta_cache = os.path.join(
            directorio_cache,
            f"evaluar_grid_search_{hash_line}.json"
        )

        if forzar_reentrenamiento and os.path.exists(ruta_cache):
            print(f"[evaluar_grid_search] Cache existente en '{ruta_cache}': se reescribira (forzar_reentrenamiento=True).")
            os.remove(ruta_cache)

        elif not forzar_reentrenamiento and os.path.exists(ruta_cache):
            print(f"[evaluar_grid_search] Cache existente en '{ruta_cache}': se cargan los resultados guardados.")
            with open(ruta_cache, "r", encoding="utf-8") as archivo_cache:
                cache_payload = json.load(archivo_cache)

            cv_results_df = pd.DataFrame(cache_payload["cv_results"])
            mejores_parametros = cache_payload["mejores_parametros"]

            mejor_estimador = clone(modelo)
            mejor_estimador.set_params(**mejores_parametros)
            mejor_estimador.fit(X, y)

            return cv_results_df, mejor_estimador, mejores_parametros

        else:
            print(f"[evaluar_grid_search] Sin cache previo en '{ruta_cache}': se entrena desde cero.")

    grid = GridSearchCV(
        estimator=clone(modelo),
        param_grid=param_grid,
        scoring=scoring,
        refit=refit,
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=True,
    )
    grid.fit(X, y)

    cv_results_df = pd.DataFrame(grid.cv_results_)
    mejores_parametros = grid.best_params_
    mejor_estimador = grid.best_estimator_

    if permitir_persistencia:
        cache_payload = {
            "cv_results": json.loads(cv_results_df.to_json(orient="records")),
            "mejores_parametros": mejores_parametros,
        }
        with open(ruta_cache, "w", encoding="utf-8") as archivo_cache:
            json.dump(cache_payload, archivo_cache, indent=2)

    return cv_results_df, mejor_estimador, mejores_parametros


# ---------------------------------------------------------------------------
# Ejecucion del kernel cuantico (migrado de funciones_nexus.py).
# Construye/envia/consulta/guarda las matrices K del QSVM, en el simulador
# local (guppy/Selene) o en Quantinuum Nexus. Solo requiere qnexus/guppy en
# tiempo de ejecucion de los backends remotos; el import es liviano (la
# autenticacion ocurre en conectar_nexus, no al importar).
# ---------------------------------------------------------------------------


MATRIX_BACKEND_OPTIONS = [
    "local_selene_statevector",
    "nexus_selene_statevector",
    "H1-1LE",
    "H1-Emulator",
    "H2-1LE",
    "H2-Emulator",
    "Helios-1E-lite",
]


def conectar_nexus(project_name):
    """
    Autentica contra Quantinuum Nexus y recupera el proyecto con ese
    nombre exacto, dejandolo como proyecto activo.

    Usa `projects.get_or_create`: si el nombre ya existe lo reutiliza: si
    no, lo crea. Por eso nunca falla con `ZeroMatches` al poner un
    `PROJECT_NAME` nuevo.

    Parameters
    ----------
    project_name : str
        Nombre exacto del proyecto en Nexus.

    Returns
    -------
    project
        Referencia al proyecto activo en Nexus.
    """
    qnx.login()
    project = qnx.projects.get_or_create(name=project_name)
    qnx.context.set_active_project(project)
    return project


def descargar_resultados_job(job_ref):
    """
    Valida que un job este COMPLETED y descarga todos sus resultados.

    La validacion consulta el estado en vivo con `qnx.jobs.status(job_ref)`
    en lugar de leer el atributo cacheado `job_ref.last_status`. Esto es
    necesario porque `last_status` es una foto del momento en que se creo
    la referencia (por ejemplo, un job recien enviado nace como SUBMITTED
    y ese atributo no se refresca solo), lo que provocaba un falso
    "job no COMPLETED" aunque en Nexus el job ya hubiera terminado.

    Antes de pedir los resultados tambien se refresca la referencia
    completa con `qnx.jobs.get(id=...)`, para no reutilizar metadatos
    cacheados del submit.

    Los conteos se extraen con `collated_counts()` (resultados de
    Selene/guppy) o, si el resultado no lo ofrece, con `get_counts()`
    (BackendResult de pytket, que es lo que devuelven H1/H2).

    Parameters
    ----------
    job_ref
        Referencia al job de Nexus.

    Returns
    -------
    tuple
        (result_refs, downloaded_results, counts_list, result_ids), donde
        `counts_list` es la lista de conteos por resultado y `result_ids`
        sus identificadores como texto.

    Raises
    ------
    RuntimeError
        Si el job no esta COMPLETED o no tiene resultados descargables.
    TypeError
        Si un resultado descargado no ofrece collated_counts() ni
        get_counts().
    """
    estado_actual = qnx.jobs.status(job_ref).status
    if estado_actual != qnx.jobs.JobStatusEnum.COMPLETED:
        raise RuntimeError(
            f"El job aun no esta COMPLETED (estado actual: {estado_actual}); "
            "no tiene resultados finales."
        )

    fresh_job_ref = qnx.jobs.get(id=job_ref.id)
    result_refs = list(qnx.jobs.results(fresh_job_ref))
    if not result_refs:
        raise RuntimeError(
            f"El job {job_ref.id} ya figura COMPLETED, pero Nexus aun no "
            "publica resultados descargables. Vuelve a consultar mas tarde."
        )

    downloaded_results = [ref.download_result() for ref in result_refs]
    counts_list = []
    for resultado in downloaded_results:
        if hasattr(resultado, "collated_counts"):
            counts_list.append(resultado.collated_counts())
        elif hasattr(resultado, "get_counts"):
            counts_list.append(resultado.get_counts())
        else:
            raise TypeError(
                "El resultado descargado no ofrece collated_counts() ni get_counts()."
            )
    result_ids = [str(ref.id) for ref in result_refs]

    return result_refs, downloaded_results, counts_list, result_ids


def compilar_y_subir_hugr(circuito, nombre):
    """
    Compila un circuito de guppy a HUGR y lo sube a Nexus.

    Parameters
    ----------
    circuito
        Funcion decorada con `@guppy`.
    nombre : str
        Nombre con el que se registra el HUGR en Nexus.

    Returns
    -------
    tuple
        (hugr_binary, ref_hugr), el paquete compilado y su referencia
        subida a Nexus.
    """
    hugr_binary = circuito.compile()
    ref_hugr = qnx.hugr.upload(hugr_package=hugr_binary, name=nombre)
    return hugr_binary, ref_hugr


def ejecutar_local(circuito, n_qubits=2, n_shots=100, seed=42, simulator="statevector"):
    """
    Ejecuta un circuito de guppy en el emulador local (Selene) y devuelve
    el resultado y sus conteos.

    Parameters
    ----------
    circuito
        Funcion decorada con `@guppy`.
    n_qubits : int, optional
        Numero de qubits del emulador. Por defecto 2.
    n_shots : int, optional
        Numero de repeticiones. Por defecto 100.
    seed : int, optional
        Semilla para reproducibilidad. Por defecto 42.
    simulator : str, optional
        "statevector" (por defecto) o "stabilizer". El statevector es
        necesario para circuitos con rotaciones arbitrarias (como el
        kernel ZZ); el stabilizer solo simula programas puramente
        Clifford, aunque escala a mas qubits.

    Returns
    -------
    tuple
        (result, counts), el resultado local y sus `collated_counts()`.
        Se usa `collated_counts()` (no `register_counts()`) para que el
        formato de los conteos coincida con el de los jobs de Nexus:
        un diccionario plano ``{tupla_de_pares_(registro, valor): count}``.

    Raises
    ------
    ValueError
        Si `simulator` no es "statevector" ni "stabilizer".
    """
    emulator = (
        circuito
        .emulator(n_qubits=n_qubits)
        .with_shots(n_shots)
        .with_seed(seed)
    )

    if simulator == "statevector":
        emulator = emulator.statevector_sim()
    elif simulator == "stabilizer":
        emulator = emulator.stabilizer_sim()
    else:
        raise ValueError('simulator debe ser "statevector" o "stabilizer"')

    result = emulator.run()
    counts = result.collated_counts()
    return result, counts


def kernel_circuit(x_i, x_j, feature_map="zz", remove_barriers=False):
    """
    Construye el circuito del kernel ``U(x_j)^dagger U(x_i)`` para el
    feature map indicado.

    La probabilidad de medir ``00...0`` en este circuito estima el kernel
    de fidelidad ``K(x_i, x_j) = |<phi(x_j)|phi(x_i)>|^2``, cualquiera sea
    el feature map elegido.

    Parameters
    ----------
    x_i, x_j : array-like
        Vectores de features de las dos observaciones a comparar.
    feature_map : str or callable, optional
        Feature map a usar: "zz" (default), "zyy" o "ry_cx_rx", o una
        funcion ``f(x) -> pytket.Circuit``.
    remove_barriers : bool, optional
        Si True, elimina las barreras (necesario antes de cargar el
        circuito en guppy o enviarlo a un backend). Por defecto False,
        para conservarlas en la inspeccion visual.

    Returns
    -------
    pytket.Circuit
        Circuito del kernel para el par (x_i, x_j).
    """
    fmap = obtener_feature_map(feature_map)

    circuit_xi = fmap(x_i)
    circuit_xj_adjoint = fmap(x_j).dagger()

    kernel = circuit_xi.copy()
    kernel.append(circuit_xj_adjoint)

    if remove_barriers:
        RemoveBarriers().apply(kernel)

    return kernel


def resumen_kernel_desde_resultado(execution_result, n_qubits):
    """
    Extrae el conteo del estado ``00...0`` de un resultado de ejecucion,
    que es lo unico que participa en el kernel de fidelidad.

    Soporta los dos tipos de resultado del flujo:

    - Guppy/Selene: expone ``register_counts()`` y los conteos del kernel
      estan bajo la etiqueta ``"kernel_measurement"``.
    - BackendResult de pytket (H1/H2): expone ``get_counts()`` con
      outcomes en tuplas de bits.

    Parameters
    ----------
    execution_result
        Resultado devuelto por el emulador local o descargado de Nexus.
    n_qubits : int
        Numero de qubits del circuito (define el estado cero esperado).

    Returns
    -------
    dict
        Con claves: zero_state, zero_count, shots, kernel_rate
        (``zero_count / shots``, sin redondear: se conserva el valor
        crudo tal como sale del backend).

    Raises
    ------
    TypeError
        Si el resultado no ofrece register_counts() ni get_counts().
    """
    zero_state = "0" * n_qubits

    if hasattr(execution_result, "register_counts"):
        # Guppy/Selene: conteos agrupados por etiqueta de result().
        register_counts = execution_result.register_counts()
        kernel_counts = register_counts["kernel_measurement"]
        shots = int(sum(kernel_counts.values()))
        zero_count = int(kernel_counts.get(zero_state, 0))

    elif hasattr(execution_result, "get_counts"):
        # Circuitos pytket en H1/H2: Counter con outcomes de bits.
        kernel_counts = execution_result.get_counts()
        shots = int(sum(kernel_counts.values()))
        zero_count = 0

        for outcome, count in kernel_counts.items():
            try:
                is_zero = all(int(bit) == 0 for bit in outcome)
            except TypeError:
                is_zero = str(outcome).replace(" ", "") in {
                    zero_state,
                    f"({','.join('0' for _ in range(n_qubits))})",
                }
            if is_zero:
                zero_count += int(count)

    else:
        raise TypeError(
            "Tipo de resultado no soportado: se esperaba un resultado "
            "Guppy con register_counts() o BackendResult con get_counts()."
        )

    kernel_rate = zero_count / shots if shots else 0.0
    return {
        "zero_state": zero_state,
        "zero_count": zero_count,
        "shots": shots,
        "kernel_rate": kernel_rate,
    }


def crear_programa_kernel_guppy(x_i, x_j, feature_map="zz"):
    """
    Crea el programa guppy ejecutable para un par del kernel, conservando
    la definicion del circuito en pytket.

    El circuito ``U(x_j)^dagger U(x_i)`` se construye en pytket (sin
    barreras), se carga en guppy con `guppy.load_pytket` y se envuelve en
    un programa que reserva los qubits, aplica el kernel y mide todo bajo
    la etiqueta ``"kernel_measurement"``.

    Parameters
    ----------
    x_i, x_j : array-like
        Vectores de features de las dos observaciones a comparar.
    feature_map : str or callable, optional
        Feature map a usar ("zz", "zyy" o "ry_cx_rx"). Por defecto "zz".

    Returns
    -------
    tuple
        (pair_program, pair_circuit): el programa guppy verificado con
        ``.check()`` y el circuito pytket subyacente.
    """
    pair_circuit = kernel_circuit(
        x_i,
        x_j,
        feature_map=feature_map,
        remove_barriers=True,
    )
    pair_n_qubits = pair_circuit.n_qubits

    pair_kernel = guppy.load_pytket(
        "pair_kernel",
        pair_circuit,
        use_arrays=True,
    )

    @guppy
    def pair_program() -> None:
        qs = array(qubit() for _ in range(comptime(pair_n_qubits)))
        pair_kernel(qs)
        measurements = measure_array(qs)
        result("kernel_measurement", measurements)

    pair_program.check()
    return pair_program, pair_circuit


def ejecutar_kernel_guppy(
    x_i,
    x_j,
    n_shots=1000,
    seed=42,
    simulator="statevector",
    feature_map="zz",
):
    """
    Ejecuta un unico ``K(i, j)`` del kernel en guppy/Selene local.

    Parameters
    ----------
    x_i, x_j : array-like
        Vectores de features de las dos observaciones a comparar.
    n_shots : int, optional
        Numero de repeticiones. Por defecto 1000.
    seed : int, optional
        Semilla para reproducibilidad. Por defecto 42.
    simulator : str, optional
        Simulador local ("statevector" o "stabilizer"). Por defecto
        "statevector", necesario para las rotaciones del kernel ZZ.
    feature_map : str or callable, optional
        Feature map a usar ("zz", "zyy" o "ry_cx_rx"). Por defecto "zz".

    Returns
    -------
    dict
        Con claves: kernel (la tasa K(i,j)), summary, counts, program,
        pytket_circuit y raw_result.
    """
    pair_program, pair_circuit = crear_programa_kernel_guppy(
        x_i, x_j, feature_map=feature_map
    )
    pair_result, pair_counts = ejecutar_local(
        pair_program,
        n_qubits=pair_circuit.n_qubits,
        n_shots=n_shots,
        seed=seed,
        simulator=simulator,
    )
    summary = resumen_kernel_desde_resultado(
        pair_result,
        pair_circuit.n_qubits,
    )

    return {
        "kernel": summary["kernel_rate"],
        "summary": summary,
        "counts": pair_counts,
        "program": pair_program,
        "pytket_circuit": pair_circuit,
        "raw_result": pair_result,
    }


def construir_matriz_kernel_guppy(
    X,
    row_labels=None,
    n_shots=1000,
    seed=42,
    simulator="statevector",
    ejecutar_diagonal=False,
    feature_map="zz",
):
    """
    Construye la matriz kernel simetrica ejecutando un programa guppy por
    par en el simulador local.

    Solo se ejecuta el triangulo superior; cada valor se refleja por
    simetria (``K(i,j) = K(j,i)``). La diagonal puede fijarse en 1 sin
    ejecutar circuitos, ya que idealmente ``K(x_i, x_i) = 1``.

    Parameters
    ----------
    X : pandas.DataFrame or array-like
        Matriz (n_filas, n_features) con las observaciones.
    row_labels : list, optional
        Etiquetas de las filas (por ejemplo, sus indices originales en
        train). Por defecto 0..n-1.
    n_shots : int, optional
        Shots por circuito. Por defecto 1000.
    seed : int, optional
        Semilla base; cada par usa ``seed + i * n + j``. Por defecto 42.
    simulator : str, optional
        Simulador local. Por defecto "statevector".
    ejecutar_diagonal : bool, optional
        Si True, tambien ejecuta los pares (i, i); si False, fija la
        diagonal en 1. Por defecto False.
    feature_map : str or callable, optional
        Feature map a usar ("zz", "zyy" o "ry_cx_rx"). Por defecto "zz".

    Returns
    -------
    dict
        Con claves: kernel_matrix (numpy.ndarray), run_summary
        (DataFrame con una fila por circuito), row_labels, n_circuits y
        n_shots_per_circuit.

    Raises
    ------
    ValueError
        Si X no es bidimensional o row_labels no coincide en longitud.
    """
    if isinstance(X, pd.DataFrame):
        X_values = X.to_numpy(dtype=float)
    else:
        X_values = np.asarray(X, dtype=float)

    if X_values.ndim != 2:
        raise ValueError("X debe tener forma (n_filas, n_features).")

    n_samples = X_values.shape[0]
    if row_labels is None:
        row_labels = list(range(n_samples))
    else:
        row_labels = list(row_labels)

    if len(row_labels) != n_samples:
        raise ValueError("row_labels debe tener una etiqueta por fila de X.")

    matrix = np.zeros((n_samples, n_samples), dtype=float)
    run_rows = []

    if ejecutar_diagonal:
        total_circuits = n_samples * (n_samples + 1) // 2
    else:
        np.fill_diagonal(matrix, 1.0)
        total_circuits = n_samples * (n_samples - 1) // 2

    with tqdm(
        total=total_circuits,
        desc="Matriz kernel con Guppy",
        unit="circuito",
    ) as progress:
        for i in range(n_samples):
            start_j = i if ejecutar_diagonal else i + 1

            for j in range(start_j, n_samples):
                pair_seed = seed + i * n_samples + j
                pair = ejecutar_kernel_guppy(
                    X_values[i],
                    X_values[j],
                    n_shots=n_shots,
                    seed=pair_seed,
                    simulator=simulator,
                    feature_map=feature_map,
                )

                value = pair["kernel"]
                matrix[i, j] = value
                matrix[j, i] = value

                run_rows.append({
                    "matrix_i": i,
                    "matrix_j": j,
                    "row_i": row_labels[i],
                    "row_j": row_labels[j],
                    "n_qubits": X_values.shape[1],
                    "zero_state": pair["summary"]["zero_state"],
                    "zero_count": pair["summary"]["zero_count"],
                    "shots": pair["summary"]["shots"],
                    "kernel_rate": value,
                    "seed": pair_seed,
                })

                progress.set_postfix({
                    "rows": f"{row_labels[i]},{row_labels[j]}",
                    "Kij": f"{value:.4f}",
                })
                progress.update(1)

    return {
        "kernel_matrix": matrix,
        "run_summary": pd.DataFrame(run_rows),
        "row_labels": row_labels,
        "n_circuits": total_circuits,
        "n_shots_per_circuit": n_shots,
    }


def guardar_matriz_kernel_run(
    matrix_result,
    source="local_statevector",
    run_id=None,
    job_id=None,
    job_name=None,
    directory="data/runs",
):
    """
    Guarda en CSV el resumen compacto de una construccion de matriz
    kernel (una fila por circuito ejecutado).

    El CSV usa siempre el mismo esquema de columnas, sin importar si la
    matriz se construyo en el simulador local o se reconstruyo desde un
    job remoto: el bloque de identificacion (run_id, source, job_id,
    job_name) seguido de las columnas del par y su resultado. Las
    columnas que no aplican a un origen (por ejemplo `seed` en remoto, o
    `result_id`/`backend`/`program_format` en local) quedan vacias, de
    modo que todos los archivos coinciden en formato desde el origen.
    Separador ";" como el resto de los CSV del proyecto.

    Parameters
    ----------
    matrix_result : dict
        Resultado devuelto por `construir_matriz_kernel_guppy` (o el
        equivalente reconstruido desde un job remoto).
    source : str, optional
        Origen de la ejecucion. Por defecto "local_statevector".
    run_id : optional
        Identificador de la ejecucion; si no se pasa, se genera uno
        local nuevo (``matrix-local-<uuid>``).
    job_id : optional
        Identificador del job de Nexus, si aplica.
    job_name : str, optional
        Nombre del job de Nexus, si aplica.
    directory : str, optional
        Carpeta destino. Por defecto "data/runs".

    Returns
    -------
    pathlib.Path
        Ruta del CSV generado (``kernel_matrix_run_<run_id>.csv``).
    """
    if run_id is None:
        run_id = f"matrix-local-{uuid.uuid4().hex[:12]}"

    run_df = matrix_result["run_summary"].copy()
    run_df.insert(0, "job_name", "" if job_name is None else str(job_name))
    run_df.insert(0, "job_id", "" if job_id is None else str(job_id))
    run_df.insert(0, "source", source)
    run_df.insert(0, "run_id", str(run_id))

    columnas = [
        "run_id", "source", "job_id", "job_name",
        "matrix_i", "matrix_j", "row_i", "row_j", "result_id",
        "backend", "program_format", "n_qubits", "zero_state",
        "zero_count", "shots", "kernel_rate", "seed",
    ]
    run_df = run_df.reindex(columns=columnas, fill_value="")

    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_run_id = str(run_id).replace("/", "_").replace("\\", "_")
    output_path = output_dir / f"kernel_matrix_run_{safe_run_id}.csv"
    run_df.to_csv(output_path, index=False, sep=";")
    return output_path


def guardar_kernel_qsvm(
    matrix_result,
    source,
    job_id=None,
    job_name=None,
    run_id=None,
    directory="data/runs",
):
    """
    Guarda la matriz de Gram del kernel en formato cuadrado, lista para
    usarse como kernel precomputado en un SVM (``SVC(kernel="precomputed")``),
    junto con un CSV de metadatos que documenta su procedencia.

    A diferencia de `guardar_matriz_kernel_run` (formato largo, una fila
    por circuito para inspeccion), aqui se persiste el artefacto final: la
    matriz cuadrada K etiquetada por observacion. Genera dos archivos con
    separador ";":

    - ``kernel_qsvm_<run_id>.csv``: la matriz K, con filas y columnas
      etiquetadas por los indices de las observaciones (`row_labels`).
      Se guarda con los valores crudos, sin redondear.
    - ``kernel_qsvm_<run_id>_meta.csv``: una fila con la procedencia
      (source/backend, job_id, shots, numero de filas, qubits, etc.).

    Parameters
    ----------
    matrix_result : dict
        Resultado devuelto por `construir_matriz_kernel_guppy` o
        reconstruido por `consultar_matriz_nexus`.
    source : str
        Origen de la ejecucion (por ejemplo "local_statevector" o
        "nexus_H1-1LE").
    job_id, job_name : optional
        Identificadores del job de Nexus, si aplica.
    run_id : optional
        Identificador de la matriz; si no se pasa, se usa el job_id o se
        genera uno local nuevo. El nombre del archivo se deriva de el, asi
        que reguardar la misma matriz remota reemplaza el archivo.
    directory : str, optional
        Carpeta destino. Por defecto "data/runs".

    Returns
    -------
    tuple of pathlib.Path
        (ruta_matriz, ruta_meta).
    """
    from datetime import datetime, timezone

    if run_id is None:
        run_id = str(job_id) if job_id is not None else f"kernel-local-{uuid.uuid4().hex[:12]}"

    K = np.asarray(matrix_result["kernel_matrix"], dtype=float)
    labels = list(matrix_result["row_labels"])
    # Para K_test la matriz es rectangular (test x train): las columnas
    # llevan sus propias etiquetas. Para K_train (cuadrada) col == row.
    col_labels = list(matrix_result.get("col_labels", labels))
    es_rectangular = (len(col_labels) != len(labels)) or (col_labels != labels)
    matrix_kind = "test" if es_rectangular else "train"
    run_summary = matrix_result.get("run_summary")

    def _primer_valor(col, default=""):
        if run_summary is not None and col in run_summary.columns and len(run_summary):
            return run_summary[col].iloc[0]
        return default

    meta = {
        "run_id": str(run_id),
        "source": source,
        "matrix_kind": matrix_kind,
        "job_id": "" if job_id is None else str(job_id),
        "job_name": "" if job_name is None else str(job_name),
        "backend": _primer_valor("backend"),
        "program_format": _primer_valor("program_format"),
        "n_rows": len(labels),
        "rows": ",".join(str(x) for x in labels),
        "n_cols": len(col_labels),
        "cols": ",".join(str(x) for x in col_labels),
        "n_qubits": _primer_valor("n_qubits"),
        "shots_per_circuit": matrix_result.get("n_shots_per_circuit", ""),
        "n_circuits": matrix_result.get("n_circuits", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_run_id = str(run_id).replace("/", "_").replace("\\", "_")

    # K_train conserva el nombre historico kernel_qsvm_<run_id>.csv;
    # K_test lleva el infijo _test_ para no pisar la matriz de train.
    infijo = "test_" if es_rectangular else ""
    ruta_matriz = output_dir / f"kernel_qsvm_{infijo}{safe_run_id}.csv"
    pd.DataFrame(K, index=labels, columns=col_labels).to_csv(ruta_matriz, sep=";")

    ruta_meta = output_dir / f"kernel_qsvm_{infijo}{safe_run_id}_meta.csv"
    pd.DataFrame([meta]).to_csv(ruta_meta, index=False, sep=";")

    return ruta_matriz, ruta_meta


def cargar_kernel_qsvm(ruta_matriz):
    """
    Carga una matriz de Gram guardada por `guardar_kernel_qsvm`.

    Parameters
    ----------
    ruta_matriz : str or pathlib.Path
        Ruta del CSV de la matriz (``kernel_qsvm_<run_id>.csv``).

    Returns
    -------
    pandas.DataFrame
        Matriz cuadrada con filas y columnas etiquetadas por observacion.
        Para un SVM, usar ``.to_numpy()`` con ``SVC(kernel="precomputed")``.
    """
    return pd.read_csv(ruta_matriz, sep=";", index_col=0)


def preparar_backend_matriz(backend, n_qubits):
    """
    Valida el backend de la matriz y construye su configuracion.

    Parameters
    ----------
    backend : str
        Uno de `MATRIX_BACKEND_OPTIONS`.
    n_qubits : int
        Numero de qubits de los circuitos (features del dataset).

    Returns
    -------
    dict
        Con claves: execution_target ("local" o "nexus"), nexus_target,
        program_format ("hugr" o "pytket_circuit") y backend_config.
        Para el backend local, todo salvo execution_target es None.

    Raises
    ------
    ValueError
        Si el backend no esta en `MATRIX_BACKEND_OPTIONS`.
    """
    if backend not in MATRIX_BACKEND_OPTIONS:
        raise ValueError(
            f"Backend desconocido: {backend}. Opciones: {MATRIX_BACKEND_OPTIONS}"
        )

    if backend == "local_selene_statevector":
        return {
            "execution_target": "local",
            "nexus_target": None,
            "program_format": None,
            "backend_config": None,
        }

    nexus_target = (
        "selene_statevector" if backend == "nexus_selene_statevector" else backend
    )

    if nexus_target == "selene_statevector":
        program_format = "hugr"
        backend_config = qnx.models.SeleneConfig(
            n_qubits=n_qubits,
            simulator=qnx.models.StatevectorSimulator(),
        )
    elif nexus_target.startswith("Helios-"):
        program_format = "hugr"
        backend_config = qnx.models.HeliosConfig(system_name=nexus_target)
    else:
        # H1/H2 no aceptan HUGR directo. Se usa el Circuit de pytket.
        program_format = "pytket_circuit"
        backend_config = qnx.models.QuantinuumConfig(device_name=nexus_target)

    return {
        "execution_target": "nexus",
        "nexus_target": nexus_target,
        "program_format": program_format,
        "backend_config": backend_config,
    }


def enviar_matriz_kernel_nexus(
    matrix_input,
    rows,
    backend_info,
    backend,
    n_shots,
    seed,
    ejecutar_diagonal,
    feature_map="zz",
):
    """
    Sube un programa por par del triangulo de la matriz y envia el job a
    Nexus: execute job directo para formato HUGR (Selene/Helios) o
    compile job para circuitos pytket (H1/H2, que no aceptan HUGR; el
    execute se encadena despues en `consultar_matriz_nexus`).

    Parameters
    ----------
    matrix_input : pandas.DataFrame
        Filas de train seleccionadas (solo features).
    rows : list of int
        Indices originales de las filas.
    backend_info : dict
        Resultado de `preparar_backend_matriz`.
    backend : str
        Nombre del backend (para el registro).
    n_shots : int
        Shots por circuito.
    seed : int
        Semilla base (solo registro; los backends remotos no la usan).
    ejecutar_diagonal : bool
        Si True, incluye los pares (i, i).
    feature_map : str or callable, optional
        Feature map a usar ("zz", "zyy" o "ry_cx_rx"). Por defecto "zz".

    Returns
    -------
    dict
        Estado de la matriz remota: backend, nexus_target,
        program_format, backend_config, n_shots, rows, n_qubits,
        ejecutar_diagonal, pair_metadata, job_ref y compile_job_ref.
    """
    matrix_values = matrix_input.to_numpy(dtype=float)
    n_rows = len(rows)
    program_format = backend_info["program_format"]
    nexus_target = backend_info["nexus_target"]
    suffix = uuid.uuid4().hex[:8]
    # Etiqueta del feature map para nombrar jobs/circuitos en Nexus.
    map_tag = feature_map if isinstance(feature_map, str) else "custom"

    pair_metadata = []
    hugr_refs = []
    circuit_refs = []
    planned = (
        n_rows * (n_rows + 1) // 2 if ejecutar_diagonal
        else n_rows * (n_rows - 1) // 2
    )

    with tqdm(
        total=planned,
        desc=f"Preparando {program_format}",
        unit="circuito",
    ) as progress:
        for i in range(n_rows):
            start_j = i if ejecutar_diagonal else i + 1

            for j in range(start_j, n_rows):
                pair_name = f"{map_tag}-matrix-{rows[i]}-{rows[j]}-{suffix}"
                metadata = {
                    "matrix_i": i,
                    "matrix_j": j,
                    "row_i": rows[i],
                    "row_j": rows[j],
                    "seed": seed + i * n_rows + j,
                }

                if program_format == "hugr":
                    pair_program, _ = crear_programa_kernel_guppy(
                        matrix_values[i],
                        matrix_values[j],
                        feature_map=feature_map,
                    )
                    _, pair_ref = compilar_y_subir_hugr(pair_program, pair_name)
                    hugr_refs.append(pair_ref)
                else:
                    pair_circuit = kernel_circuit(
                        matrix_values[i],
                        matrix_values[j],
                        feature_map=feature_map,
                        remove_barriers=True,
                    )
                    pair_circuit.measure_all()
                    pair_ref = qnx.circuits.upload(
                        circuit=pair_circuit,
                        name=pair_name,
                    )
                    circuit_refs.append(pair_ref)

                pair_metadata.append(metadata)
                progress.update(1)

    estado = {
        "backend": backend,
        "nexus_target": nexus_target,
        "program_format": program_format,
        "backend_config": backend_info["backend_config"],
        "n_shots": n_shots,
        "rows": list(rows),
        "n_qubits": matrix_input.shape[1],
        "ejecutar_diagonal": ejecutar_diagonal,
        "pair_metadata": pair_metadata,
        "job_ref": None,
        "compile_job_ref": None,
    }

    if program_format == "hugr":
        estado["job_ref"] = qnx.start_execute_job(
            programs=hugr_refs,
            n_shots=[n_shots] * len(hugr_refs),
            backend_config=estado["backend_config"],
            name=f"{map_tag}-kernel-matrix-{nexus_target}-{suffix}",
        )
        print("Job de ejecucion HUGR enviado.")
        print("Execute Job ID:", estado["job_ref"].id)
    else:
        estado["compile_job_ref"] = qnx.start_compile_job(
            programs=circuit_refs,
            backend_config=estado["backend_config"],
            optimisation_level=2,
            skip_intermediate_circuits=True,
            name=f"compile-{map_tag}-matrix-{nexus_target}-{suffix}",
        )
        print("Circuitos pytket subidos; compile job enviado.")
        print("Compile Job ID:", estado["compile_job_ref"].id)

    print("Backend:", backend)
    print("Formato:", program_format)
    print("Programas:", len(pair_metadata))
    print("Consulta el avance con la celda de consulta; no reenvies esta celda.")
    return estado


def iniciar_matriz_kernel(
    train_df,
    rows,
    backend,
    run_matrix,
    n_shots=1000,
    seed=42,
    ejecutar_diagonal=False,
    guardar=True,
    project_name=None,
    feature_map="zz",
):
    """
    Punto de entrada de la seccion de matriz kernel: valida la seleccion
    y despacha segun el interruptor y el backend.

    - ``run_matrix=False``: solo imprime el plan (filas, backend y
      cuantos circuitos se ejecutarian) sin consumir shots.
    - Backend local: construye la matriz en Selene local y devuelve el
      resultado (opcionalmente guardado en CSV).
    - Backend de Nexus: sube los programas, envia el job y devuelve el
      estado para seguirlo con `consultar_matriz_nexus`.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Particion de train con solo features.
    rows : list of int
        Indices de las filas que forman la matriz.
    backend : str
        Uno de `MATRIX_BACKEND_OPTIONS`.
    run_matrix : bool
        Interruptor de seguridad; en False solo imprime el plan.
    n_shots : int, optional
        Shots por circuito. Por defecto 1000.
    seed : int, optional
        Semilla base local. Por defecto 42.
    ejecutar_diagonal : bool, optional
        Si True, ejecuta tambien los pares (i, i); si False, fija la
        diagonal en 1. Por defecto False.
    guardar : bool, optional
        Si True (default), guarda el run local en CSV.
    project_name : str, optional
        Proyecto de Nexus (requerido para backends remotos).
    feature_map : str or callable, optional
        Feature map a usar ("zz", "zyy" o "ry_cx_rx"). Por defecto "zz".

    Returns
    -------
    tuple
        (estado, matrix_result): el estado remoto (o None) y el
        resultado local (o None).

    Raises
    ------
    ValueError
        Si rows es invalido, el backend no existe o falta project_name
        en un backend remoto.
    IndexError
        Si algun indice de rows queda fuera de train.
    """
    if not rows:
        raise ValueError("MATRIX_ROWS no puede estar vacio.")
    if min(rows) < 0 or max(rows) >= len(train_df):
        raise IndexError("Algun indice de MATRIX_ROWS queda fuera de train.")
    if len(set(rows)) != len(rows):
        raise ValueError("MATRIX_ROWS no debe contener indices repetidos.")

    matrix_input = train_df.iloc[rows]
    n_rows = len(rows)
    planned = (
        n_rows * (n_rows + 1) // 2 if ejecutar_diagonal
        else n_rows * (n_rows - 1) // 2
    )

    backend_info = preparar_backend_matriz(backend, matrix_input.shape[1])

    if not run_matrix:
        print("Construccion de matriz desactivada.")
        print("Filas seleccionadas:", rows)
        print("Backend seleccionado:", backend)
        print("Opciones:", MATRIX_BACKEND_OPTIONS)
        print("Circuitos que se ejecutarian:", planned)
        print("Cambia RUN_MATRIX = True cuando quieras iniciar.")
        return None, None

    if backend_info["execution_target"] == "local":
        matrix_result = construir_matriz_kernel_guppy(
            X=matrix_input,
            row_labels=rows,
            n_shots=n_shots,
            seed=seed,
            simulator="statevector",
            ejecutar_diagonal=ejecutar_diagonal,
            feature_map=feature_map,
        )
        if guardar:
            ruta = guardar_matriz_kernel_run(
                matrix_result, source="local_statevector"
            )
            print("Run de matriz guardado en:", ruta)
        return None, matrix_result

    if project_name is None:
        raise ValueError("Se requiere project_name para backends de Nexus.")
    conectar_nexus(project_name)
    estado = enviar_matriz_kernel_nexus(
        matrix_input,
        rows,
        backend_info,
        backend,
        n_shots=n_shots,
        seed=seed,
        ejecutar_diagonal=ejecutar_diagonal,
        feature_map=feature_map,
    )
    return estado, None


def iniciar_matriz_kernel_test(
    train_df,
    rows,
    test_df,
    backend,
    run_matrix,
    test_rows=None,
    n_shots=1000,
    seed=42,
    guardar=True,
    project_name=None,
    feature_map="zz",
):
    """
    Construye la matriz kernel de test ``K_test = K(X_test, X_train)``,
    el kernel rectangular (n_test x m) que un QSVM usa para predecir.

    Reutiliza la maquinaria de matriz cuadrada: apila ``[X_test, X_train]``
    (test primero), construye la conjunta simetrica
    ``K([X_test, X_train], [X_test, X_train])`` con los mismos
    constructores que K_train y recorta el bloque test x train. El precio
    de reutilizar esa maquinaria es que tambien se computan los bloques
    test-test y train-train, que se descartan.

    El bloque extraido es puramente fuera de la diagonal de la conjunta,
    asi que la diagonal nunca se ejecuta (equivale a `ejecutar_diagonal`
    False, fijado internamente).

    Debe usarse el **mismo `feature_map`** con el que se construyo K_train:
    de lo contrario ambas matrices no serian comparables y el SVM quedaria
    inconsistente.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Particion de train con solo features.
    rows : list of int
        Indices de las filas de train que forman las columnas de K_test
        (las mismas que definieron K_train).
    test_df : pandas.DataFrame
        Particion de test con solo features.
    backend : str
        Uno de `MATRIX_BACKEND_OPTIONS`.
    run_matrix : bool
        Interruptor de seguridad; en False solo imprime el plan.
    test_rows : list of int, optional
        Indices de las filas de test que forman las filas de K_test. Por
        defecto, todas las filas de test.
    n_shots : int, optional
        Shots por circuito. Por defecto 1000.
    seed : int, optional
        Semilla base local. Por defecto 42.
    guardar : bool, optional
        Si True (default), guarda el run local en CSV.
    project_name : str, optional
        Proyecto de Nexus (requerido para backends remotos).
    feature_map : str or callable, optional
        Feature map a usar ("zz", "zyy" o "ry_cx_rx"). Por defecto "zz".
        Debe coincidir con el de K_train.

    Returns
    -------
    tuple
        (estado, matrix_result): el estado remoto (o None) y el resultado
        local de K_test (o None). El estado remoto lleva ``kind="test"``
        para que `consultar_matriz_nexus` recorte el bloque al completarse.

    Raises
    ------
    ValueError
        Si rows/test_rows es invalido, el backend no existe o falta
        project_name en un backend remoto.
    IndexError
        Si algun indice queda fuera de su particion.
    """
    if not rows:
        raise ValueError("MATRIX_ROWS no puede estar vacio.")
    if min(rows) < 0 or max(rows) >= len(train_df):
        raise IndexError("Algun indice de MATRIX_ROWS queda fuera de train.")
    if len(set(rows)) != len(rows):
        raise ValueError("MATRIX_ROWS no debe contener indices repetidos.")

    if test_rows is None:
        test_rows = list(range(len(test_df)))
    if not test_rows:
        raise ValueError("TEST_ROWS no puede estar vacio.")
    if min(test_rows) < 0 or max(test_rows) >= len(test_df):
        raise IndexError("Algun indice de TEST_ROWS queda fuera de test.")
    if len(set(test_rows)) != len(test_rows):
        raise ValueError("TEST_ROWS no debe contener indices repetidos.")

    train_input = train_df.iloc[rows].reset_index(drop=True)
    test_input = test_df.iloc[test_rows].reset_index(drop=True)
    n_test = len(test_input)
    n_train = len(train_input)

    # Test primero, train despues: asi K_test queda en el bloque superior
    # derecho de la conjunta.
    matrix_input = pd.concat([test_input, train_input], ignore_index=True)
    joint_labels = list(range(len(matrix_input)))

    backend_info = preparar_backend_matriz(backend, matrix_input.shape[1])

    if not run_matrix:
        print("Construccion de K_test desactivada.")
        print("Forma del resultado:", (n_test, n_train))
        print("Filas de test:", test_rows)
        print("Columnas de train:", rows)
        print("Backend seleccionado:", backend)
        print("Opciones:", MATRIX_BACKEND_OPTIONS)
        pares_utiles = n_test * n_train
        circuitos_conjunta = (n_test + n_train) * (n_test + n_train - 1) // 2
        print("Pares test x train necesarios:", pares_utiles)
        print(
            "Circuitos de la conjunta (incluye test-test y train-train):",
            circuitos_conjunta,
        )
        print("Cambia RUN_MATRIX = True cuando quieras iniciar.")
        return None, None

    if backend_info["execution_target"] == "local":
        full_result = construir_matriz_kernel_guppy(
            X=matrix_input,
            row_labels=joint_labels,
            n_shots=n_shots,
            seed=seed,
            simulator="statevector",
            ejecutar_diagonal=False,
            feature_map=feature_map,
        )
        matrix_result = _extraer_bloque_test(
            full_result, n_test, test_rows, rows
        )
        if guardar:
            ruta = guardar_matriz_kernel_run(
                matrix_result, source="local_statevector_test"
            )
            print("Run de K_test guardado en:", ruta)
        return None, matrix_result

    if project_name is None:
        raise ValueError("Se requiere project_name para backends de Nexus.")
    conectar_nexus(project_name)
    estado = enviar_matriz_kernel_nexus(
        matrix_input,
        joint_labels,
        backend_info,
        backend,
        n_shots=n_shots,
        seed=seed,
        ejecutar_diagonal=False,
        feature_map=feature_map,
    )
    # Metadatos para que consultar_matriz_nexus recorte el bloque test x train.
    estado["kind"] = "test"
    estado["n_test"] = n_test
    estado["test_rows"] = list(test_rows)
    estado["train_rows"] = list(rows)
    print(
        "Nexus construira la conjunta",
        (n_test + n_train, n_test + n_train),
        "; al consultar se recorta K_test",
        (n_test, n_train),
    )
    return estado, None


def _extraer_bloque_test(full_result, n_test, test_rows, train_rows):
    """
    Recorta el bloque test x train de una matriz kernel conjunta.

    Dada la matriz conjunta ``K([X_test, X_train], [X_test, X_train])``
    (con las filas de test primero), extrae el bloque superior derecho
    ``K[:n_test, n_test:]`` = ``K(X_test, X_train)``, que es el kernel
    rectangular que consume el QSVM para predecir. Es agnostico al feature
    map: opera solo sobre la matriz numerica ya calculada.

    Parameters
    ----------
    full_result : dict
        Resultado de la matriz conjunta (mismo esquema que
        `construir_matriz_kernel_guppy`).
    n_test : int
        Numero de filas de test (las primeras de la conjunta).
    test_rows, train_rows : list
        Indices originales de test y train, para etiquetar filas/columnas.

    Returns
    -------
    dict
        Mismo esquema que `construir_matriz_kernel_guppy` pero rectangular:
        kernel_matrix (n_test x m), run_summary filtrado a los pares
        test x train, row_labels (test), col_labels (train), n_circuits y
        n_shots_per_circuit.
    """
    K_full = np.asarray(full_result["kernel_matrix"], dtype=float)
    n_train = len(train_rows)
    K_block = K_full[:n_test, n_test:n_test + n_train].copy()

    run_summary = full_result.get("run_summary")
    if run_summary is not None and len(run_summary):
        # En el triangulo superior de la conjunta, los pares test x train
        # son exactamente los que tienen matrix_i en test y matrix_j en train.
        mask = (
            (run_summary["matrix_i"] < n_test)
            & (run_summary["matrix_j"] >= n_test)
        )
        bloque_summary = run_summary.loc[mask].copy()
        # Reetiquetar a los indices reales de test (fila) y train (columna).
        bloque_summary["row_i"] = bloque_summary["matrix_i"].map(
            lambda k: test_rows[int(k)]
        )
        bloque_summary["row_j"] = bloque_summary["matrix_j"].map(
            lambda k: train_rows[int(k) - n_test]
        )
        n_circuits = len(bloque_summary)
    else:
        bloque_summary = run_summary
        n_circuits = 0

    return {
        "kernel_matrix": K_block,
        "run_summary": bloque_summary,
        "row_labels": list(test_rows),
        "col_labels": list(train_rows),
        "n_circuits": n_circuits,
        "n_shots_per_circuit": full_result.get("n_shots_per_circuit", ""),
    }


def consultar_matriz_nexus(estado, guardar=True):
    """
    Consulta el avance de una matriz kernel remota sin bloquear el
    cuaderno. Reejecutar esta consulta es seguro; nunca reenvia los
    programas.

    Para H1/H2 primero refresca el compile job y, cuando termina,
    encadena automaticamente el execute job con los circuitos
    compilados. Despues consulta el execute job y, al aparecer
    COMPLETED, descarga los resultados y reconstruye la matriz.

    Parameters
    ----------
    estado : dict or None
        Estado devuelto por `iniciar_matriz_kernel` (None si no se
        envio una matriz remota).
    guardar : bool, optional
        Si True (default), guarda el run en CSV al completarse.

    Returns
    -------
    tuple
        (estado, matrix_result): el estado actualizado (con el execute
        job encadenado si aplico) y el resultado reconstruido, o None si
        el job sigue en curso.

    Raises
    ------
    RuntimeError
        Si el compile o el execute job terminan sin exito, o si la
        cantidad de circuitos/resultados no coincide con los pares.
    """
    terminal_errors = {"ERROR", "CANCELLED", "TERMINATED", "DEPLETED"}

    if estado is None:
        print("No hay una matriz remota nueva que consultar.")
        return estado, None

    if estado["compile_job_ref"] is not None and estado["job_ref"] is None:
        compile_status = qnx.jobs.status(estado["compile_job_ref"])
        print("Compile Job ID:", estado["compile_job_ref"].id)
        print("Compile status:", compile_status.status)
        print("Compile message:", compile_status.message)

        if compile_status.status == qnx.jobs.JobStatusEnum.COMPLETED:
            fresh_compile_ref = qnx.jobs.get(id=estado["compile_job_ref"].id)
            compile_results = list(qnx.jobs.results(fresh_compile_ref))
            compiled_refs = [item.get_output() for item in compile_results]

            if len(compiled_refs) != len(estado["pair_metadata"]):
                raise RuntimeError(
                    "La cantidad de circuitos compilados no coincide con los pares."
                )

            estado["job_ref"] = qnx.start_execute_job(
                programs=compiled_refs,
                n_shots=[estado["n_shots"]] * len(compiled_refs),
                backend_config=estado["backend_config"],
                name=(
                    f"execute-zz-matrix-{estado['nexus_target']}-"
                    f"{uuid.uuid4().hex[:8]}"
                ),
            )
            print("Compilacion terminada; execute job enviado.")
            print("Execute Job ID:", estado["job_ref"].id)
        elif compile_status.status.value in terminal_errors:
            raise RuntimeError(f"El compile job termino sin exito: {compile_status}")
        else:
            print("La compilacion continua. Reejecuta esta celda mas tarde.")

    if estado["job_ref"] is None:
        if estado["compile_job_ref"] is not None:
            print("Aun no existe execute job; espera a que termine la compilacion.")
        return estado, None

    matrix_status = qnx.jobs.status(estado["job_ref"])
    print("Execute Job ID:", estado["job_ref"].id)
    print("Execute status:", matrix_status.status)
    print("Execute message:", matrix_status.message)

    queue_position = getattr(matrix_status, "queue_position", None)
    if queue_position is not None:
        print("Posicion en cola:", queue_position)

    if matrix_status.status != qnx.jobs.JobStatusEnum.COMPLETED:
        if matrix_status.status.value in terminal_errors:
            raise RuntimeError(f"El execute job termino sin exito: {matrix_status}")
        print(
            "El execute job aun no esta COMPLETED (estado consultado en vivo). "
            "Reejecuta esta celda mas tarde; confirma que este Execute Job ID sea "
            "el mismo que figura COMPLETED en Nexus (en H1/H2 el compile job es un "
            "job aparte y termina antes que el execute)."
        )
        return estado, None

    _, matrix_downloaded, _, matrix_result_ids = descargar_resultados_job(
        estado["job_ref"]
    )
    if len(matrix_downloaded) != len(estado["pair_metadata"]):
        raise RuntimeError(
            "La cantidad de resultados no coincide con los pares enviados."
        )

    n_rows = len(estado["rows"])
    remote_matrix = np.eye(n_rows, dtype=float)
    if estado["ejecutar_diagonal"]:
        remote_matrix.fill(0.0)
    remote_run_rows = []

    for metadata, downloaded_result, result_id in zip(
        estado["pair_metadata"],
        matrix_downloaded,
        matrix_result_ids,
    ):
        summary = resumen_kernel_desde_resultado(
            downloaded_result,
            estado["n_qubits"],
        )
        i = metadata["matrix_i"]
        j = metadata["matrix_j"]
        remote_matrix[i, j] = summary["kernel_rate"]
        remote_matrix[j, i] = summary["kernel_rate"]

        remote_run_rows.append({
            **metadata,
            "result_id": result_id,
            "backend": estado["backend"],
            "program_format": estado["program_format"],
            "n_qubits": estado["n_qubits"],
            "zero_state": summary["zero_state"],
            "zero_count": summary["zero_count"],
            "shots": summary["shots"],
            "kernel_rate": summary["kernel_rate"],
        })

    matrix_result = {
        "kernel_matrix": remote_matrix,
        "run_summary": pd.DataFrame(remote_run_rows),
        "row_labels": estado["rows"],
        "n_circuits": len(estado["pair_metadata"]),
        "n_shots_per_circuit": estado["n_shots"],
    }

    # Para un job de K_test se reconstruye la conjunta y se recorta el
    # bloque test x train; para K_train se devuelve la matriz completa.
    es_test = estado.get("kind") == "test"
    if es_test:
        matrix_result = _extraer_bloque_test(
            matrix_result,
            estado["n_test"],
            estado["test_rows"],
            estado["train_rows"],
        )
        source = f"nexus_{estado['backend']}_test"
        print("Matriz K_test reconstruida (bloque test x train recortado).")
    else:
        source = f"nexus_{estado['backend']}"
        print("Matriz kernel reconstruida.")

    if guardar:
        ruta = guardar_matriz_kernel_run(
            matrix_result,
            source=source,
            run_id=str(estado["job_ref"].id),
            job_id=estado["job_ref"].id,
            job_name=getattr(estado["job_ref"].annotations, "name", None),
        )
        print("Run de matriz guardado en:", ruta)

    return estado, matrix_result


def _cargar_kernels(familias, destino, cual):
    """Carga K_train o K_test ya calculadas y reporta el estado por familia.

    Motor comun de `cargar_ktrain`/`cargar_ktest`. Para cada familia lee la
    ruta correspondiente (`ruta_train` si ``cual="train"``, `ruta_test` si
    ``cual="test"``) con `cargar_kernel_qsvm` y la guarda en `destino`
    (clave ``"{feature_map}_{tamano}"``). No usa widgets ni dependencias de
    frontend: el estado se devuelve como un DataFrame que se muestra igual
    en cualquier entorno.

    Parameters
    ----------
    familias : list
        Lista de entradas de FAMILIAS (2 o 4 elementos).
    destino : dict
        Diccionario (se rellena in place) familia -> DataFrame de la matriz.
    cual : {"train", "test"}
        Cual de las dos rutas de la familia cargar.

    Returns
    -------
    pandas.DataFrame
        Estado por familia, con columnas: "familia", "estado"
        ("cargada" / "pendiente" / "error"), "n_{cual}", "ruta_{cual}" y
        "detalle".
    """
    if cual not in {"train", "test"}:
        raise ValueError(f"cual debe ser 'train' o 'test'; recibido {cual!r}.")

    col_n = f"n_{cual}"
    col_ruta = f"ruta_{cual}"
    filas = []
    for familia in familias:
        feature_map, tamano, ruta_train, ruta_test = parsear_familia(familia)
        clave = f"{feature_map}_{tamano}"
        ruta = ruta_train if cual == "train" else ruta_test

        if not ruta:
            filas.append({
                "familia": clave, "estado": "pendiente", col_n: "",
                col_ruta: "", "detalle": f"sin ruta_{cual}; usar calcular_k{cual}",
            })
            continue

        try:
            K = cargar_kernel_qsvm(ruta)
            valores = K.to_numpy(dtype=float)
            if not np.isfinite(valores).all():
                raise ValueError("la matriz contiene NaN o infinitos")
            if valores.min() < -1e-12 or valores.max() > 1 + 1e-12:
                raise ValueError("la matriz contiene valores fuera del rango [0, 1]")
            if cual == "train":
                if K.shape != (tamano, tamano):
                    raise ValueError(
                        f"forma {K.shape}; se esperaba ({tamano}, {tamano})"
                    )
                if not np.allclose(valores, valores.T, atol=1e-12, rtol=0):
                    raise ValueError("K_train no es simetrica")
                if not np.allclose(np.diag(valores), 1.0, atol=1e-12, rtol=0):
                    raise ValueError("la diagonal de K_train no es unitaria")
            elif K.shape[1] != tamano:
                raise ValueError(
                    f"forma {K.shape}; K_test debe tener {tamano} columnas"
                )
            destino[clave] = K
            filas.append({
                "familia": clave, "estado": "cargada", col_n: K.shape[0],
                col_ruta: ruta, "detalle": f"{K.shape[0]}x{K.shape[1]}",
            })
        except Exception as error:
            filas.append({
                "familia": clave, "estado": "error", col_n: "",
                col_ruta: ruta, "detalle": str(error),
            })

    return pd.DataFrame(filas)


def cargar_ktrain(familias, k_train_por_familia):
    """Carga las matrices K_train ya calculadas y reporta el estado por familia.

    Recorre `familias` (ver `parsear_familia`) y carga la `ruta_train` de
    cada una en `k_train_por_familia`. Las familias sin `ruta_train` quedan
    marcadas como pendientes (se calculan con `calcular_ktrain`). Con un
    "run all" deja listas todas las familias ya calculadas sin interaccion.

    Parameters
    ----------
    familias : list
        Lista de entradas de FAMILIAS (2 o 4 elementos).
    k_train_por_familia : dict
        Diccionario (se rellena in place) familia -> DataFrame de K_train.

    Returns
    -------
    pandas.DataFrame
        Estado por familia (ver `_cargar_kernels`).
    """
    return _cargar_kernels(familias, k_train_por_familia, "train")


def cargar_ktest(familias, k_test_por_familia):
    """Carga las matrices K_test ya calculadas y reporta el estado por familia.

    Espejo de `cargar_ktrain` para la `ruta_test` de cada familia.

    Parameters
    ----------
    familias : list
        Lista de entradas de FAMILIAS (2 o 4 elementos).
    k_test_por_familia : dict
        Diccionario (se rellena in place) familia -> DataFrame de K_test.

    Returns
    -------
    pandas.DataFrame
        Estado por familia (ver `_cargar_kernels`).
    """
    return _cargar_kernels(familias, k_test_por_familia, "test")


def _train_df_familia(muestreo_df, variables, tamano, nivel_por_tamano):
    """Extrae las filas de train (features) de una familia segun su nivel de muestreo."""
    nivel = nivel_muestreo(tamano, nivel_por_tamano)
    mask = (muestreo_df["_PartInd_"] == 0) & (muestreo_df["_Muestreo_"] >= nivel)
    train_df = muestreo_df.loc[mask, variables].reset_index(drop=True)
    return train_df, list(range(len(train_df)))


def calcular_ktrain(
    familia,
    muestreo_df,
    variables,
    nivel_por_tamano,
    backend,
    n_shots,
    seed,
    ejecutar_diagonal,
    project_name,
    k_train_por_familia,
):
    """Calcula (o envia) la matriz K_train de una familia pendiente.

    Construye las filas de train de la familia segun su nivel de muestreo y
    llama a `iniciar_matriz_kernel` con `backend`:

    - Backend **local** (sincrono): construye la matriz en el momento, la
      persiste con `guardar_kernel_qsvm`, la carga en `k_train_por_familia`
      y devuelve None (ya no hay nada que consultar).
    - Backend **Nexus** (asincrono): sube el job y devuelve el `estado`;
      cuando el job termine, pasar ese `estado` a `consultar_ktrain` para
      guardar y cargar la matriz.

    Parameters
    ----------
    familia : tuple
        Entrada de FAMILIAS (2 o 4 elementos).
    muestreo_df : pandas.DataFrame
        Hoja `muestreos`, con las features escaladas y `_PartInd_`/`_Muestreo_`.
    variables : list of str
        Columnas que alimentan el kernel.
    nivel_por_tamano : dict
        Mapeo tamano -> nivel de muestreo.
    backend : str
        Backend de ejecucion (uno de `MATRIX_BACKEND_OPTIONS`).
    n_shots, seed : int
        Shots por circuito y semilla base.
    ejecutar_diagonal : bool
        Si True ejecuta tambien la diagonal (i, i).
    project_name : str
        Proyecto de Nexus (solo backends remotos).
    k_train_por_familia : dict
        Diccionario (se rellena in place) familia -> DataFrame de K_train.

    Returns
    -------
    dict or None
        `estado` del job de Nexus (para `consultar_ktrain`), o None si el
        backend local ya calculo y guardo la matriz.
    """
    feature_map, tamano, _ruta_train, _ruta_test = parsear_familia(familia)
    clave = f"{feature_map}_{tamano}"
    train_df, rows = _train_df_familia(muestreo_df, variables, tamano, nivel_por_tamano)

    estado, matrix_result = iniciar_matriz_kernel(
        train_df, rows, backend, True,
        n_shots=n_shots, seed=seed, ejecutar_diagonal=ejecutar_diagonal,
        guardar=False, project_name=project_name, feature_map=feature_map,
    )

    if estado is not None:
        # Nexus: job enviado; se consulta luego.
        print(f"[{clave}] Job enviado a Nexus; usa consultar_ktrain(estado, ...) cuando termine.")
        return estado

    # Local: la matriz ya esta construida.
    ruta, _meta = guardar_kernel_qsvm(matrix_result, source="local_statevector_train")
    k_train_por_familia[clave] = cargar_kernel_qsvm(ruta)
    print(f"[{clave}] K_train calculada y guardada en: {ruta}")
    return None


def consultar_ktrain(estado, familia, backend, k_train_por_familia):
    """Consulta un job de K_train en Nexus; guarda y carga la matriz si esta lista.

    Complemento de `calcular_ktrain` para backends Nexus (asincronos).
    Consulta el job de `estado`; si aun no termina, avisa y no cambia nada
    (se puede reintentar). Si ya termino, persiste la matriz cuadrada con
    `guardar_kernel_qsvm` y la carga en `k_train_por_familia`.

    Parameters
    ----------
    estado : dict
        Estado devuelto por `calcular_ktrain` (job de Nexus).
    familia : tuple
        Entrada de FAMILIAS de esa matriz (para la clave del diccionario).
    backend : str
        Backend usado (para etiquetar la procedencia).
    k_train_por_familia : dict
        Diccionario (se rellena in place) familia -> DataFrame de K_train.

    Returns
    -------
    bool
        True si la matriz quedo guardada y cargada; False si el job aun no
        esta listo (reintentar mas tarde).
    """
    feature_map, tamano, _ruta_train, _ruta_test = parsear_familia(familia)
    clave = f"{feature_map}_{tamano}"

    estado, matrix_remoto = consultar_matriz_nexus(estado, guardar=False)
    if matrix_remoto is None:
        print(f"[{clave}] El job aun no esta listo; reintenta mas tarde.")
        return False

    ruta, _meta = guardar_kernel_qsvm(
        matrix_remoto, source=f"nexus_{backend}_train", job_id=estado["job_ref"].id
    )
    k_train_por_familia[clave] = cargar_kernel_qsvm(ruta)
    print(f"[{clave}] K_train guardada y cargada desde: {ruta}")
    return True



def _test_df_familia(muestreo_df, variables, tamano, nivel_por_tamano):
    """Extrae las filas de test (features) de una familia segun su nivel de muestreo."""
    nivel = nivel_muestreo(tamano, nivel_por_tamano)
    mask = (muestreo_df["_PartInd_"] == 1) & (muestreo_df["_Muestreo_"] >= nivel)
    test_df = muestreo_df.loc[mask, variables].reset_index(drop=True)
    return test_df, list(range(len(test_df)))


def calcular_ktest(
    familia,
    muestreo_df,
    variables,
    nivel_por_tamano,
    backend,
    n_shots,
    seed,
    project_name,
    k_test_por_familia,
):
    """Calcula (o envia) la matriz K_test de una familia pendiente.

    Espejo de `calcular_ktrain` para la matriz rectangular
    ``K_test = K(X_test, X_train)`` (filas = test, columnas = train). Es un
    job independiente del de K_train: puede lanzarse sin esperar al de
    train, siempre con el **mismo feature map** de la familia. La diagonal
    no aplica aqui (el bloque test x train queda fuera de la diagonal de la
    matriz conjunta, `iniciar_matriz_kernel_test` lo maneja internamente).

    - Backend **local** (sincrono): construye la matriz en el momento, la
      persiste con `guardar_kernel_qsvm`, la carga en `k_test_por_familia`
      y devuelve None.
    - Backend **Nexus** (asincrono): sube el job y devuelve el `estado`;
      cuando el job termine, pasar ese `estado` a `consultar_ktest`.

    Parameters
    ----------
    familia : tuple
        Entrada de FAMILIAS (2 o 4 elementos).
    muestreo_df : pandas.DataFrame
        Hoja `muestreos`, con las features escaladas y `_PartInd_`/`_Muestreo_`.
    variables : list of str
        Columnas que alimentan el kernel.
    nivel_por_tamano : dict
        Mapeo tamano -> nivel de muestreo.
    backend : str
        Backend de ejecucion (uno de `MATRIX_BACKEND_OPTIONS`).
    n_shots, seed : int
        Shots por circuito y semilla base.
    project_name : str
        Proyecto de Nexus (solo backends remotos).
    k_test_por_familia : dict
        Diccionario (se rellena in place) familia -> DataFrame de K_test.

    Returns
    -------
    dict or None
        `estado` del job de Nexus (para `consultar_ktest`), o None si el
        backend local ya calculo y guardo la matriz.
    """
    feature_map, tamano, _ruta_train, _ruta_test = parsear_familia(familia)
    clave = f"{feature_map}_{tamano}"
    train_df, rows = _train_df_familia(muestreo_df, variables, tamano, nivel_por_tamano)
    test_df, test_rows = _test_df_familia(muestreo_df, variables, tamano, nivel_por_tamano)

    estado, matrix_result = iniciar_matriz_kernel_test(
        train_df, rows, test_df, backend, True,
        test_rows=test_rows, n_shots=n_shots, seed=seed,
        guardar=False, project_name=project_name, feature_map=feature_map,
    )

    if estado is not None:
        # Nexus: job enviado; se consulta luego.
        print(f"[{clave}] Job de K_test enviado a Nexus; usa consultar_ktest(estado, ...) cuando termine.")
        return estado

    # Local: la matriz ya esta construida.
    ruta, _meta = guardar_kernel_qsvm(matrix_result, source="local_statevector_test")
    k_test_por_familia[clave] = cargar_kernel_qsvm(ruta)
    print(f"[{clave}] K_test calculada y guardada en: {ruta}")
    return None


def consultar_ktest(estado, familia, backend, k_test_por_familia):
    """Consulta un job de K_test en Nexus; guarda y carga la matriz si esta lista.

    Complemento de `calcular_ktest` para backends Nexus (asincronos). El
    `estado` lleva ``kind="test"``, asi que `consultar_matriz_nexus` recorta
    automaticamente el bloque test x train al completarse el job.

    Parameters
    ----------
    estado : dict
        Estado devuelto por `calcular_ktest` (job de Nexus).
    familia : tuple
        Entrada de FAMILIAS de esa matriz (para la clave del diccionario).
    backend : str
        Backend usado (para etiquetar la procedencia).
    k_test_por_familia : dict
        Diccionario (se rellena in place) familia -> DataFrame de K_test.

    Returns
    -------
    bool
        True si la matriz quedo guardada y cargada; False si el job aun no
        esta listo (reintentar mas tarde).
    """
    feature_map, tamano, _ruta_train, _ruta_test = parsear_familia(familia)
    clave = f"{feature_map}_{tamano}"

    estado, matrix_remoto = consultar_matriz_nexus(estado, guardar=False)
    if matrix_remoto is None:
        print(f"[{clave}] El job de K_test aun no esta listo; reintenta mas tarde.")
        return False

    ruta, _meta = guardar_kernel_qsvm(
        matrix_remoto, source=f"nexus_{backend}_test", job_id=estado["job_ref"].id
    )
    k_test_por_familia[clave] = cargar_kernel_qsvm(ruta)
    print(f"[{clave}] K_test guardada y cargada desde: {ruta}")
    return True


# ---------------------------------------------------------------------------
# Paso 3: resultados por familia (evaluacion QSVM, graficos y estadisticos).
# ---------------------------------------------------------------------------

def graficar_matriz_kernel(K, titulo="Matriz del kernel cuantico", mostrar_valores=False, ax=None):
    """Grafica una matriz kernel (cuadrada o rectangular) como mapa de calor.

    Cada celda representa la fidelidad ``K(x_i, x_j)`` entre un par de
    observaciones, en el rango [0, 1] (vmin/vmax fijos para comparar
    matrices de distintas corridas con la misma escala de color).

    Parameters
    ----------
    K : array-like or pandas.DataFrame
        Matriz kernel, cuadrada (K_train) o rectangular (K_test).
    titulo : str, optional
        Titulo del grafico.
    mostrar_valores : bool, optional
        Si True, anota el valor en cada celda. Por defecto False.
    ax : matplotlib.axes.Axes or None, optional
        Ejes donde dibujar; si es None crea figura y ejes nuevos.

    Returns
    -------
    matplotlib.axes.Axes
        Los ejes con el mapa de calor.
    """
    K = np.asarray(K, dtype=float)
    if K.ndim != 2:
        raise ValueError("K debe ser una matriz bidimensional.")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        K, annot=mostrar_valores, fmt=".2f", vmin=0, vmax=1, square=True,
        cbar_kws={"label": "Fidelidad"}, ax=ax,
    )
    ax.set_xlabel("Observacion de train")
    ax.set_ylabel("Observacion" + (" de train" if K.shape[0] == K.shape[1] else " de test"))
    ax.set_title(titulo)
    return ax


def graficar_matriz_confusion_ax(y_true, y_pred, ax, titulo="Matriz de confusion",
                                 display_labels=("No potable", "Potable")):
    """Grafica una unica matriz de confusion sobre un `ax` dado.

    A diferencia de `graficar_matrices_confusion` (que arma su propia figura
    de dos paneles), dibuja una sola matriz sobre ejes ya existentes, para
    componerla con otros graficos en una misma figura.

    Parameters
    ----------
    y_true, y_pred : array-like
        Etiquetas reales y predichas.
    ax : matplotlib.axes.Axes
        Ejes donde dibujar.
    titulo : str, optional
        Titulo del panel.
    display_labels : tuple of str, optional
        Nombres de las clases. Por defecto ("No potable", "Potable").

    Returns
    -------
    matplotlib.axes.Axes
        Los ejes con la matriz de confusion.
    """
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=display_labels, cmap="Blues",
        ax=ax, colorbar=True, text_kw={"fontsize": 11},
    )
    ax.set_title(titulo, fontsize=12, fontweight="bold")
    ax.set_xlabel("Prediccion", fontsize=10)
    ax.set_ylabel("Real", fontsize=10)
    ax.grid(False)
    return ax


def etiquetas_familia(muestreo_df, tamano, nivel_por_tamano, objetivo="Potability"):
    """Recupera y_train/y_test de la hoja `muestreos` para el nivel de una familia.

    El orden posicional (``reset_index(drop=True)``) reproduce exactamente el
    usado al construir las matrices kernel, para que las etiquetas alineen
    con las filas/columnas de K.

    Parameters
    ----------
    muestreo_df : pandas.DataFrame
        Hoja `muestreos`, con `objetivo`, `_PartInd_` y `_Muestreo_`.
    tamano : int
        Tamano de train de la familia (16 o 32).
    nivel_por_tamano : dict
        Mapeo tamano -> nivel de muestreo.
    objetivo : str, optional
        Columna objetivo. Por defecto "Potability".

    Returns
    -------
    tuple of pandas.Series
        (y_train, y_test), enteras.
    """
    nivel = nivel_muestreo(tamano, nivel_por_tamano)
    train_mask = (muestreo_df["_PartInd_"] == 0) & (muestreo_df["_Muestreo_"] >= nivel)
    test_mask = (muestreo_df["_PartInd_"] == 1) & (muestreo_df["_Muestreo_"] >= nivel)
    y_train = muestreo_df.loc[train_mask, objetivo].reset_index(drop=True).astype(int)
    y_test = muestreo_df.loc[test_mask, objetivo].reset_index(drop=True).astype(int)
    return y_train, y_test


def evaluar_familia_qsvm(
    K_train,
    y_train,
    K_test,
    y_test,
    param_grid=None,
    random_state=42,
    n_jobs=-1,
):
    """Entrena un SVM de kernel precomputado con busqueda de C y evalua train/test.

    Ajusta ``SVC(kernel="precomputed", class_weight="balanced")`` buscando el
    mejor `C` por validacion cruzada estratificada (refit por f1), reentrena
    con ese `C` y calcula las metricas de `classification_metrics` en train y
    test. `K_train` es la matriz de Gram cuadrada; `K_test` es rectangular
    (test x train).

    Parameters
    ----------
    K_train : array-like
        Matriz de Gram cuadrada de train.
    y_train : array-like
        Etiquetas de train.
    K_test : array-like
        Matriz rectangular test x train.
    y_test : array-like
        Etiquetas de test.
    param_grid : dict or None, optional
        Grid de `C`. Por defecto ``{"C": [1e-6, 0.1, 1, 10]}``.
    random_state : int, optional
        Semilla del SVC y del CV. Por defecto 42.

    Returns
    -------
    dict
        Con claves: "mejor_C", "mejor_f1_cv", "metrics_train",
        "metrics_test", "y_pred_train", "y_pred_test".
    """
    if param_grid is None:
        param_grid = {"C": [1e-6, 0.1, 1, 10]}

    K_train = np.asarray(K_train, dtype=float)
    K_test = np.asarray(K_test, dtype=float)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    if K_train.ndim != 2 or K_train.shape[0] != K_train.shape[1]:
        raise ValueError(f"K_train debe ser cuadrada; forma recibida {K_train.shape}.")
    if K_test.ndim != 2 or K_test.shape[1] != K_train.shape[0]:
        raise ValueError(
            "K_test debe tener una columna por fila de K_train; "
            f"formas recibidas K_train={K_train.shape}, K_test={K_test.shape}."
        )
    if len(y_train) != K_train.shape[0] or len(y_test) != K_test.shape[0]:
        raise ValueError(
            "Las etiquetas no coinciden con las matrices: "
            f"len(y_train)={len(y_train)}, len(y_test)={len(y_test)}, "
            f"K_train={K_train.shape}, K_test={K_test.shape}."
        )
    if not np.isfinite(K_train).all() or not np.isfinite(K_test).all():
        raise ValueError("Las matrices kernel contienen NaN o infinitos.")
    if not np.allclose(K_train, K_train.T, atol=1e-12, rtol=0):
        raise ValueError("K_train debe ser simetrica.")
    if K_train.min() < -1e-12 or K_test.min() < -1e-12:
        raise ValueError("Las matrices kernel contienen valores menores que 0.")
    if K_train.max() > 1 + 1e-12 or K_test.max() > 1 + 1e-12:
        raise ValueError("Las matrices kernel contienen valores mayores que 1.")

    qsvm = SVC(kernel="precomputed", class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    grid = GridSearchCV(
        estimator=qsvm, param_grid=param_grid,
        scoring={"accuracy": "accuracy", "precision": "precision",
                 "recall": "recall", "balanced_accuracy": "balanced_accuracy",
                 "f1": "f1"},
        refit="f1", cv=cv, n_jobs=n_jobs, return_train_score=True,
    )
    grid.fit(K_train, y_train)
    modelo = grid.best_estimator_

    y_pred_train = modelo.predict(K_train)
    y_score_train = modelo.decision_function(K_train)
    metrics_train = classification_metrics(y_train, y_pred_train, y_score_train)

    y_pred_test = modelo.predict(K_test)
    y_score_test = modelo.decision_function(K_test)
    metrics_test = classification_metrics(y_test, y_pred_test, y_score_test)

    return {
        "mejor_C": grid.best_params_["C"],
        "mejor_f1_cv": grid.best_score_,
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
    }


def graficar_resultado_familia(K_train, y_train, y_pred_train, y_test, y_pred_test, nombre):
    """Arma la figura 1x3 de resultados de una familia.

    Panel 1: heatmap de K_train. Panel 2: matriz de confusion de train.
    Panel 3: matriz de confusion de test.

    Parameters
    ----------
    K_train : array-like
        Matriz de Gram de train (para el heatmap).
    y_train, y_pred_train : array-like
        Etiquetas reales y predichas de train.
    y_test, y_pred_test : array-like
        Etiquetas reales y predichas de test.
    nombre : str
        Nombre de la familia (titulo de la figura).

    Returns
    -------
    tuple
        (fig, axes) de matplotlib.
    """
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    graficar_matriz_kernel(K_train, titulo=f"K_train - {nombre}", ax=axes[0])
    graficar_matriz_confusion_ax(y_train, y_pred_train, ax=axes[1], titulo="Train")
    graficar_matriz_confusion_ax(y_test, y_pred_test, ax=axes[2], titulo="Test")
    fig.suptitle(nombre, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Paso 4: SVM clasico comparativo, con el mismo tamano de muestra que cada
# familia QSVM (replica la cantidad de registros, no el feature map).
# ---------------------------------------------------------------------------

def entrenar_svm_muestra(
    muestreo_df,
    variables,
    tamano,
    nivel_por_tamano,
    objetivo="Potability",
    param_grid=None,
    random_state=42,
    directorio_cache="cache",
):
    """Entrena un SVM clasico (RBF) sobre la muestra de un tamano dado.

    Replica la cantidad de registros de train/test que usa una familia
    QSVM de ese `tamano` (mismas filas de `muestreo_df`, mismas
    `variables`), para poder comparar el QSVM contra un baseline clasico
    entrenado con identica cantidad de datos. Un SVM clasico no depende
    del feature map, asi que basta un entrenamiento por tamano (no uno por
    familia): las familias que comparten tamano comparan contra el mismo
    resultado.

    La busqueda de hiperparametros usa `evaluar_grid_search` (mismo cache
    en disco que el resto del cuaderno), asi que entrenar el mismo tamano
    mas de una vez no recalcula nada. Las metricas se calculan con
    `classification_metrics`, igual que `evaluar_familia_qsvm`, para que
    ambos modelos sean directamente comparables.

    Parameters
    ----------
    muestreo_df : pandas.DataFrame
        Hoja `muestreos`, con las features escaladas, `objetivo`,
        `_PartInd_` y `_Muestreo_`.
    variables : list of str
        Columnas predictoras (las mismas variables seleccionadas en el
        Paso 1).
    tamano : int
        Tamano de train de la muestra (16 o 32).
    nivel_por_tamano : dict
        Mapeo tamano -> nivel de muestreo.
    objetivo : str, optional
        Columna objetivo. Por defecto "Potability".
    param_grid : dict or None, optional
        Grid de hiperparametros del SVM RBF. Por defecto
        ``{"C": [0.1, 1, 10], "gamma": ["scale", "auto", 0.01]}``.
    random_state : int, optional
        Semilla del SVC y del CV. Por defecto 42.
    directorio_cache : str, optional
        Carpeta de cache de `evaluar_grid_search`. Por defecto "cache".

    Returns
    -------
    dict
        Con claves: "mejores_parametros", "metrics_train", "metrics_test",
        "y_train", "y_test", "y_pred_train", "y_pred_test".
    """
    if param_grid is None:
        param_grid = {"C": [0.1, 1, 10], "gamma": ["scale", "auto", 0.01]}

    nivel = nivel_muestreo(tamano, nivel_por_tamano)
    train_mask = (muestreo_df["_PartInd_"] == 0) & (muestreo_df["_Muestreo_"] >= nivel)
    test_mask = (muestreo_df["_PartInd_"] == 1) & (muestreo_df["_Muestreo_"] >= nivel)

    X_train = muestreo_df.loc[train_mask, variables].reset_index(drop=True)
    y_train = muestreo_df.loc[train_mask, objetivo].reset_index(drop=True).astype(int)
    X_test = muestreo_df.loc[test_mask, variables].reset_index(drop=True)
    y_test = muestreo_df.loc[test_mask, objetivo].reset_index(drop=True).astype(int)

    modelo = SVC(kernel="rbf", class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
               "balanced_accuracy": "balanced_accuracy", "f1": "f1"}

    _cv_results, modelo_final, mejores_parametros = evaluar_grid_search(
        X_train, y_train, modelo, param_grid, cv, scoring, "f1",
        directorio_cache=directorio_cache,
    )

    y_pred_train = modelo_final.predict(X_train)
    y_score_train = modelo_final.decision_function(X_train)
    metrics_train = classification_metrics(y_train, y_pred_train, y_score_train)

    y_pred_test = modelo_final.predict(X_test)
    y_score_test = modelo_final.decision_function(X_test)
    metrics_test = classification_metrics(y_test, y_pred_test, y_score_test)

    return {
        "mejores_parametros": mejores_parametros,
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
    }


def graficar_comparacion_familia(
    y_train,
    y_pred_train_qsvm,
    y_test,
    y_pred_test_qsvm,
    y_pred_train_svm,
    y_pred_test_svm,
    nombre_familia,
    nombre_svm,
):
    """Arma la figura 1x4 de confusiones: QSVM train/test vs SVM train/test.

    `y_train`/`y_test` son compartidas por ambos modelos (mismas filas,
    misma muestra); solo las predicciones difieren.

    Parameters
    ----------
    y_train, y_test : array-like
        Etiquetas reales de train y test (comunes a ambos modelos).
    y_pred_train_qsvm, y_pred_test_qsvm : array-like
        Predicciones del QSVM de la familia.
    y_pred_train_svm, y_pred_test_svm : array-like
        Predicciones del SVM clasico del mismo tamano de muestra.
    nombre_familia : str
        Nombre de la familia QSVM (para los titulos).
    nombre_svm : str
        Nombre del baseline clasico (por ejemplo "SVM_16").

    Returns
    -------
    tuple
        (fig, axes) de matplotlib.
    """
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))
    graficar_matriz_confusion_ax(y_train, y_pred_train_qsvm, ax=axes[0],
                                 titulo=f"{nombre_familia} - Train")
    graficar_matriz_confusion_ax(y_test, y_pred_test_qsvm, ax=axes[1],
                                 titulo=f"{nombre_familia} - Test")
    graficar_matriz_confusion_ax(y_train, y_pred_train_svm, ax=axes[2],
                                 titulo=f"{nombre_svm} - Train")
    graficar_matriz_confusion_ax(y_test, y_pred_test_svm, ax=axes[3],
                                 titulo=f"{nombre_svm} - Test")
    fig.suptitle(f"{nombre_familia} vs {nombre_svm}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig, axes
