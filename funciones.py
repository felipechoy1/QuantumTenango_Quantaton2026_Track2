import hashlib
import os
import uuid

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
from sklearn.model_selection import cross_validate
from tqdm.auto import tqdm


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
    directorio_cache="cache"
):
    """
    Evalua un modelo mediante forward selection incremental, calculando
    F1, AUC y Recall promedio por validacion cruzada para cada cantidad
    de variables seleccionadas.

    Para cada k en 1..n_variables, selecciona las k mejores variables con
    `SequentialFeatureSelector` (direction="forward", scoring="f1") y
    evalua el modelo resultante con `cross_validate` sobre esas variables.
    Cuando k es igual al total de variables, se usan todas sin ejecutar
    el selector.

    Al ser un proceso costoso, el resultado se puede memorizar en disco:
    se arma una "line" (huella) con las columnas de X, su forma, el
    estimador (con sus hiperparametros) y el esquema de validacion
    cruzada, y se hashea (md5) para nombrar el archivo de cache. Si ya
    existe un resultado con ese hash, se reutiliza en vez de reentrenar.

    Parameters
    ----------
    X : pandas.DataFrame
        Variables predictoras.
    y : pandas.Series or array-like
        Variable objetivo.
    modelo : sklearn estimator
        Estimador base a clonar y evaluar en cada paso (debe implementar
        la API de scikit-learn: fit/predict).
    cv : int, cross-validation generator or iterable
        Estrategia de validacion cruzada usada tanto en el selector como
        en `cross_validate`.
    permitir_persistencia : bool, optional
        Si es True (por defecto), el resultado se guarda en
        `directorio_cache` y, si ya existe un resultado para la misma
        combinacion de X, modelo y cv, se reutiliza en lugar de
        recalcularlo.
    forzar_reentrenamiento : bool, optional
        Si es True, se borra el hash existente (si lo hay) y se vuelve a
        entrenar desde cero, sobrescribiendo el cache. Por defecto False.
    directorio_cache : str, optional
        Carpeta donde se guardan/leen los resultados persistidos. Por
        defecto "cache".

    Returns
    -------
    pandas.DataFrame
        Una fila por cada k evaluado, con las columnas:
        "numero_variables", "f1_cv", "auc_cv", "recall_cv" (promedios de
        validacion cruzada) y "variables" (lista de variables usadas).
    """
    ruta_cache = None

    if permitir_persistencia:
        line = "|".join([
            str(sorted(X.columns.tolist())),
            str(X.shape),
            str(getattr(y, "name", "y")),
            repr(modelo.get_params()),
            repr(cv)
        ])

        hash_line = hashlib.md5(line.encode("utf-8")).hexdigest()

        os.makedirs(directorio_cache, exist_ok=True)
        ruta_cache = os.path.join(
            directorio_cache,
            f"evaluar_forward_{hash_line}.pkl"
        )

        if forzar_reentrenamiento and os.path.exists(ruta_cache):
            print(f"[evaluar_forward] Cache existente en '{ruta_cache}': se reescribira (forzar_reentrenamiento=True).")
            os.remove(ruta_cache)

        elif not forzar_reentrenamiento and os.path.exists(ruta_cache):
            print(f"[evaluar_forward] Cache existente en '{ruta_cache}': se cargan los resultados guardados.")
            return pd.read_pickle(ruta_cache)

        else:
            print(f"[evaluar_forward] Sin cache previo en '{ruta_cache}': se entrena desde cero.")

    resultados = []
    total_variables = X.shape[1]

    for k in tqdm(range(1, total_variables + 1), desc="evaluar_forward"):

        # Forward selection
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

        # F1, AUC y recall promedio mediante CV
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

        resultados.append({
            "numero_variables": k,
            "f1_cv": metricas_cv["test_f1"].mean(),
            "auc_cv": metricas_cv["test_auc"].mean(),
            "recall_cv":metricas_cv["test_recall"].mean(),
            "variables": variables
        })

    resultado_df = pd.DataFrame(resultados)

    if permitir_persistencia:
        resultado_df.to_pickle(ruta_cache)

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


def _formatear_outcome(outcome):
    """
    Normaliza la clave de un conteo (outcome) a un texto uniforme, sin
    importar si viene de `collated_counts()` de Nexus (tuplas de pares
    (registro, valor)) o de `register_counts()` del simulador local
    (tuplas de bits u otras estructuras).

    Parameters
    ----------
    outcome : tuple or object
        Clave del diccionario de conteos.

    Returns
    -------
    str
        Representacion textual estable del outcome.
    """
    if isinstance(outcome, tuple):
        if outcome and all(
            isinstance(par, tuple) and len(par) == 2 for par in outcome
        ):
            return " | ".join(f"{registro}={valor}" for registro, valor in outcome)
        return "".join(str(valor) for valor in outcome)
    return str(outcome)


def guardar_ejecucion_csv(
    counts,
    source,
    job_id=None,
    job_name=None,
    result_ids=None,
    n_shots=None,
    run_id=None,
    directorio="data/runs"
):
    """
    Guarda los resultados de una ejecucion cuantica en un CSV con un
    formato unico, comun a todos los tipos de ejecucion (simulador local,
    job de Nexus reutilizado o job de Nexus recien terminado).

    El formato replica la tabla de resultados de Nexus (columnas
    result_index, result_id, outcome, count, shots, proportion) y la
    enriquece con identificadores de la ejecucion (run_id, source,
    job_id, job_name).

    Sobre el identificador de la ejecucion (`run_id`):

    - Si se pasa `run_id`, se usa tal cual.
    - Si no, y hay `job_id` (ejecuciones de Nexus), el `run_id` es el
      propio `job_id`. Como el nombre del archivo se deriva del `run_id`,
      volver a guardar el mismo job **reemplaza** el CSV anterior en vez
      de duplicarlo.
    - Si no hay `job_id` (simulador local), se genera un `run_id` unico
      nuevo (`local-<uuid>`), de modo que cada ejecucion local produce su
      propio archivo.

    Parameters
    ----------
    counts : dict or list of dict
        Conteos {outcome: count}. Puede ser un unico diccionario o una
        lista de diccionarios (uno por resultado del job).
    source : str
        Origen de la ejecucion, por ejemplo "local", "selected_nexus_job"
        o "new_nexus_job".
    job_id : optional
        Identificador del job de Nexus, si aplica.
    job_name : str, optional
        Nombre del job de Nexus, si aplica.
    result_ids : list, optional
        Identificadores de cada resultado, alineados con `counts` cuando
        este es una lista. Si no se pasan, quedan vacios.
    n_shots : int, optional
        Numero de shots solicitado. Solo informativo; los shots por fila
        se calculan sumando los conteos de cada resultado.
    run_id : optional
        Identificador de la ejecucion. Ver la nota de arriba sobre como
        se determina si no se pasa.
    directorio : str, optional
        Carpeta donde se guardan los CSV. Por defecto "data/runs".

    Returns
    -------
    str
        Ruta del archivo CSV generado.
    """
    if run_id is None:
        run_id = str(job_id) if job_id is not None else f"local-{uuid.uuid4().hex}"

    # Normaliza counts a una lista de diccionarios (uno por resultado)
    if isinstance(counts, dict):
        counts_list = [counts]
    else:
        counts_list = list(counts)

    filas = []
    for result_index, conteos in enumerate(counts_list):
        conteos = dict(conteos)
        total_shots = sum(conteos.values())

        if result_ids is not None and result_index < len(result_ids):
            result_id = str(result_ids[result_index])
        else:
            result_id = ""

        for outcome, count in conteos.items():
            filas.append({
                "run_id": str(run_id),
                "source": source,
                "job_id": "" if job_id is None else str(job_id),
                "job_name": "" if job_name is None else str(job_name),
                "result_index": result_index,
                "result_id": result_id,
                "outcome": _formatear_outcome(outcome),
                "count": count,
                "shots": total_shots,
                "proportion": count / total_shots if total_shots else 0.0,
            })

    df_run = pd.DataFrame(filas, columns=[
        "run_id", "source", "job_id", "job_name",
        "result_index", "result_id", "outcome", "count", "shots", "proportion",
    ])

    os.makedirs(directorio, exist_ok=True)
    nombre_seguro = str(run_id).replace("/", "_").replace("\\", "_")
    ruta_csv = os.path.join(directorio, f"run_{nombre_seguro}.csv")
    df_run.to_csv(ruta_csv, index=False)

    return ruta_csv