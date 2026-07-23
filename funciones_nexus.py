"""Funciones de apoyo para el cuaderno `guppy_real_sim_control.ipynb`.

Agrupa los pasos reutilizables de interaccion con Quantinuum Nexus y el
simulador local (guppy/Selene): conexion, listado y seleccion de jobs,
descarga de resultados, compilacion/envio de circuitos, ejecucion local y
guardado de resultados en CSV.

Este modulo esta pensado para el entorno Conda `quantum_space`, que es el
unico que tiene `qnexus`, `ipywidgets` y `guppylang` instalados.
"""

import os
import uuid

import ipywidgets as widgets
import pandas as pd
import qnexus as qnx


def conectar_nexus(project_name):
    """
    Autentica contra Quantinuum Nexus y recupera un proyecto existente por
    su nombre exacto, dejandolo como proyecto activo.

    Usa `projects.get`, por lo que nunca crea un proyecto nuevo: si el
    nombre no existe, la consulta falla en lugar de crear otro.

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
    project = qnx.projects.get(name=project_name)
    qnx.context.set_active_project(project)
    return project


def obtener_execution_jobs(project, page_size=100):
    """
    Recupera todos los jobs de tipo EXECUTE de un proyecto (solo lectura).

    Parameters
    ----------
    project
        Referencia al proyecto de Nexus.
    page_size : int, optional
        Tamano de pagina para la consulta. Por defecto 100.

    Returns
    -------
    list
        Lista de referencias a jobs de ejecucion.
    """
    return list(qnx.jobs.get_all(
        project=project,
        job_type=[qnx.jobs.JobType.EXECUTE],
        created_after=None,
        page_size=page_size
    ))


def resumir_job(job):
    """
    Devuelve el nombre y el estado legibles de un job.

    Parameters
    ----------
    job
        Referencia a un job de Nexus.

    Returns
    -------
    tuple of (str, str)
        (nombre, estado) del job.
    """
    nombre = getattr(job.annotations, "name", None) or str(job.id)
    estado = getattr(job.last_status, "value", str(job.last_status))
    return nombre, estado


def construir_selector_jobs(job_refs):
    """
    Imprime la lista de jobs disponibles y devuelve un desplegable para
    seleccionar uno por su indice.

    Parameters
    ----------
    job_refs : list
        Lista de referencias a jobs (por ejemplo, la de
        `obtener_execution_jobs`).

    Returns
    -------
    ipywidgets.Dropdown
        Selector con una opcion por job; su `.value` es el indice elegido.
    """
    opciones = []
    for indice, job in enumerate(job_refs):
        nombre, estado = resumir_job(job)
        print(f"[{indice}] {nombre} | {estado} | id={job.id}")
        opciones.append((f"[{indice}] {nombre} | {estado}", indice))

    return widgets.Dropdown(
        options=opciones,
        value=opciones[0][1] if opciones else None,
        description="Job:",
        disabled=not opciones,
        layout=widgets.Layout(width="700px")
    )


def descargar_resultados_job(job_ref):
    """
    Valida que un job este COMPLETED y descarga todos sus resultados.

    Parameters
    ----------
    job_ref
        Referencia al job de Nexus.

    Returns
    -------
    tuple
        (result_refs, downloaded_results, counts_list, result_ids), donde
        `counts_list` es la lista de conteos (`collated_counts`) por
        resultado y `result_ids` sus identificadores como texto.

    Raises
    ------
    RuntimeError
        Si el job no esta COMPLETED o no tiene resultados descargables.
    """
    if job_ref.last_status != qnx.jobs.JobStatusEnum.COMPLETED:
        raise RuntimeError(
            "El job seleccionado aun no esta COMPLETED y no tiene resultados finales."
        )

    result_refs = list(qnx.jobs.results(job_ref))
    if not result_refs:
        raise RuntimeError(f"El job {job_ref.id} no contiene resultados descargables.")

    downloaded_results = [ref.download_result() for ref in result_refs]
    counts_list = [resultado.collated_counts() for resultado in downloaded_results]
    result_ids = [str(ref.id) for ref in result_refs]

    return result_refs, downloaded_results, counts_list, result_ids


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


def _aplanar_conteos(conteos):
    """
    Aplana los conteos de un unico resultado a una lista de filas
    (outcome, count, shots), soportando las dos estructuras posibles:

    - Plana, de `collated_counts()` de Nexus: ``{outcome: count}`` con
      valores enteros. Los shots son el total del resultado.
    - Anidada por registro, de `register_counts()` del simulador local:
      ``{registro: Counter({bitstring: count})}``. Cada registro tiene
      sus propios shots (la suma de su Counter) y el outcome se escribe
      como ``"registro=bitstring"``.

    Parameters
    ----------
    conteos : dict
        Conteos de un unico resultado, en cualquiera de las dos formas.

    Returns
    -------
    list of tuple
        Lista de (outcome_text, count, shots).
    """
    conteos = dict(conteos)

    # Estructura anidada por registro: todos los valores son mappings
    # (Counter o dict), no enteros.
    anidada = bool(conteos) and all(hasattr(valor, "items") for valor in conteos.values())

    filas = []
    if anidada:
        for registro, counter in conteos.items():
            counter = dict(counter)
            shots_registro = sum(counter.values())
            for bits, count in counter.items():
                filas.append((f"{registro}={bits}", count, shots_registro))
    else:
        shots_total = sum(conteos.values())
        for outcome, count in conteos.items():
            filas.append((_formatear_outcome(outcome), count, shots_total))

    return filas


def construir_tabla_resultados(counts_list, result_ids=None):
    """
    Construye la tabla de resultados en el formato estandar de Nexus a
    partir de una lista de conteos.

    Parameters
    ----------
    counts_list : dict or list of dict
        Conteos {outcome: count}. Puede ser un unico diccionario o una
        lista de diccionarios (uno por resultado).
    result_ids : list, optional
        Identificadores de cada resultado, alineados con `counts_list`.

    Returns
    -------
    pandas.DataFrame
        Columnas: result_index, result_id, outcome, count, shots,
        proportion.
    """
    if isinstance(counts_list, dict):
        counts_list = [counts_list]

    filas = []
    for result_index, conteos in enumerate(counts_list):
        if result_ids is not None and result_index < len(result_ids):
            result_id = str(result_ids[result_index])
        else:
            result_id = ""

        for outcome_text, count, shots in _aplanar_conteos(conteos):
            filas.append({
                "result_index": result_index,
                "result_id": result_id,
                "outcome": outcome_text,
                "count": count,
                "shots": shots,
                "proportion": count / shots if shots else 0.0,
            })

    return pd.DataFrame(filas, columns=[
        "result_index", "result_id", "outcome", "count", "shots", "proportion",
    ])


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


def ejecutar_local(circuito, n_qubits=2, n_shots=100, seed=42):
    """
    Ejecuta un circuito de guppy en el emulador local (simulador
    estabilizador) y devuelve el resultado y sus conteos.

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

    Returns
    -------
    tuple
        (result, counts), el resultado local y sus `collated_counts()`.
        Se usa `collated_counts()` (no `register_counts()`) para que el
        formato de los conteos coincida con el de los jobs de Nexus:
        un diccionario plano ``{tupla_de_pares_(registro, valor): count}``.
    """
    result = (
        circuito
        .emulator(n_qubits=n_qubits)
        .stabilizer_sim()
        .with_shots(n_shots)
        .with_seed(seed)
        .run()
    )
    counts = result.collated_counts()
    return result, counts


def enviar_job_selene(ref_hugr, n_qubits=2, n_shots=100, nombre=None):
    """
    Envia un HUGR ya subido al simulador Selene de Nexus y devuelve la
    referencia del job (no bloquea; el job queda en cola/ejecucion).

    Parameters
    ----------
    ref_hugr
        Referencia al HUGR subido a Nexus.
    n_qubits : int, optional
        Numero de qubits fisicos. Por defecto 2.
    n_shots : int, optional
        Numero de repeticiones. Por defecto 100.
    nombre : str, optional
        Nombre del job.

    Returns
    -------
    job_ref
        Referencia al job de ejecucion enviado.
    """
    sim_config = qnx.models.SeleneConfig(
        n_qubits=n_qubits,
        simulator=qnx.models.StabilizerSimulator()
    )
    return qnx.start_execute_job(
        programs=[ref_hugr],
        n_shots=[n_shots],
        backend_config=sim_config,
        name=nombre
    )


def nuevo_run_id_local():
    """
    Genera un identificador unico para una ejecucion del simulador local.

    Returns
    -------
    str
        Identificador con formato `local-<uuid>`.
    """
    return f"local-{uuid.uuid4().hex[:12]}"


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
        if result_ids is not None and result_index < len(result_ids):
            result_id = str(result_ids[result_index])
        else:
            result_id = ""

        for outcome_text, count, shots in _aplanar_conteos(conteos):
            filas.append({
                "run_id": str(run_id),
                "source": source,
                "job_id": "" if job_id is None else str(job_id),
                "job_name": "" if job_name is None else str(job_name),
                "result_index": result_index,
                "result_id": result_id,
                "outcome": outcome_text,
                "count": count,
                "shots": shots,
                "proportion": count / shots if shots else 0.0,
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
