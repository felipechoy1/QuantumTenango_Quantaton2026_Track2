import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_validate
from tqdm.auto import tqdm


def graficar_box_hist_grid(df, features, variables_por_fila=2, num_bins=10):
    """
    Grafica boxplot e histograma de cada variable en una sola figura,
    organizados en una cuadricula de N variables por fila.

    Cada variable ocupa dos subplots contiguos (boxplot e histograma).
    El histograma incluye una barra adicional "Vacios" con el conteo de
    valores nulos de la variable.

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

        # Histograma
        frecuencias, limites, _ = ax_hist.hist(
            datos_validos,
            bins=num_bins,
            edgecolor="black",
            alpha=0.75
        )

        # Posicion de la barra de valores vacios
        ancho_bin = limites[1] - limites[0]
        posicion_vacios = limites[-1] + ancho_bin * 1.5

        ax_hist.bar(
            posicion_vacios,
            cantidad_vacios,
            width=ancho_bin,
            color="gray",
            edgecolor="black"
        )

        # Etiqueta sobre la barra de vacios
        ax_hist.text(
            posicion_vacios,
            cantidad_vacios,
            str(cantidad_vacios),
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
        ax_hist.set_ylabel("Frecuencia")

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

def evaluar_forward(X, y, modelo, cv):
    """
    Evalua un modelo mediante forward selection incremental, calculando
    F1, AUC y Recall promedio por validacion cruzada para cada cantidad
    de variables seleccionadas.

    Para cada k en 1..n_variables, selecciona las k mejores variables con
    `SequentialFeatureSelector` (direction="forward", scoring="f1") y
    evalua el modelo resultante con `cross_validate` sobre esas variables.
    Cuando k es igual al total de variables, se usan todas sin ejecutar
    el selector.

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

    Returns
    -------
    pandas.DataFrame
        Una fila por cada k evaluado, con las columnas:
        "numero_variables", "f1_cv", "auc_cv", "recall_cv" (promedios de
        validacion cruzada) y "variables" (lista de variables usadas).
    """

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

    return pd.DataFrame(resultados)