import hashlib
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from sklearn.model_selection import GridSearchCV, cross_validate
from tqdm.auto import tqdm

from pytket import Circuit
from pytket.circuit.display import get_circuit_renderer
from IPython.display import HTML, display


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
    metrica_refit="f1"
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
            str(sorted(X.columns.tolist())),
            str(X.shape),
            str(getattr(y, "name", "y")),
            repr(modelo.get_params()),
            repr(cv),
            repr(param_grid),
            metrica_refit if param_grid is not None else ""
        ])

        hash_line = hashlib.md5(line.encode("utf-8")).hexdigest()

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
                n_jobs=-1
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
                n_jobs=-1
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
                n_jobs=-1
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

    os.makedirs(os.path.dirname(ruta_excel), exist_ok=True)

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
            str(sorted(X.columns.tolist())),
            str(X.shape),
            str(getattr(y, "name", "y")),
            repr(modelo.get_params()),
            repr(param_grid),
            repr(cv),
            repr(scoring),
            refit,
        ])

        hash_line = hashlib.md5(line.encode("utf-8")).hexdigest()

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
        n_jobs=-1,
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
