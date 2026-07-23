"""Funciones de apoyo para el cuaderno `guppy_real_sim_control.ipynb`.

Agrupa los pasos reutilizables de interaccion con Quantinuum Nexus y el
simulador local (guppy/Selene): conexion, listado y seleccion de jobs,
descarga de resultados, compilacion/envio de circuitos, ejecucion local y
guardado de resultados en CSV.

Incluye tambien la logica del kernel cuantico ZZ para QSVM: feature map en
pytket, circuito U(x_j)^dagger U(x_i), puente pytket->guppy, ejecucion por
pares y construccion de la matriz kernel.

Este modulo requiere `qnexus`, `ipywidgets`, `guppylang` y `pytket`
(disponibles en los entornos Conda `quantum_space` y `qsvm`).
"""

import os
import uuid
from pathlib import Path

import ipywidgets as widgets
import numpy as np
import pandas as pd
import qnexus as qnx
from guppylang import guppy
from guppylang.std.builtins import array, comptime, result
from guppylang.std.quantum import measure_array, qubit
from pytket import Circuit
from pytket.passes import RemoveBarriers
from tqdm.auto import tqdm


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


def listar_jobs_si_corresponde(allow_new_execution, project, project_name):
    """
    Lista los jobs de ejecucion existentes solo en modo CONSULTA.

    En modo RUN NUEVO (`allow_new_execution=True`) no consulta nada,
    porque no se van a reutilizar resultados previos.

    Parameters
    ----------
    allow_new_execution : bool
        Interruptor maestro del cuaderno (`ALLOW_NEW_EXECUTION`).
    project
        Referencia al proyecto de Nexus.
    project_name : str
        Nombre del proyecto, solo para el mensaje informativo.

    Returns
    -------
    tuple
        (execution_job_refs, job_selector). En modo RUN NUEVO,
        `execution_job_refs` es una lista vacia y `job_selector` es None.
    """
    if allow_new_execution:
        print("No aplica: ALLOW_NEW_EXECUTION = True (run nuevo completo); no se consultan jobs existentes.")
        return [], None

    execution_job_refs = obtener_execution_jobs(project)
    print(f"Jobs disponibles en {project_name}: {len(execution_job_refs)}")
    job_selector = construir_selector_jobs(execution_job_refs)
    print("Selecciona un job y luego ejecuta la siguiente celda.")
    return execution_job_refs, job_selector


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


def cargar_job_seleccionado(allow_new_execution, job_selector, execution_job_refs):
    """
    Carga los resultados del job elegido en el selector solo en modo
    CONSULTA. En modo RUN NUEVO no hay resultados previos que cargar.

    Parameters
    ----------
    allow_new_execution : bool
        Interruptor maestro del cuaderno (`ALLOW_NEW_EXECUTION`).
    job_selector : ipywidgets.Dropdown or None
        Selector devuelto por `listar_jobs_si_corresponde`.
    execution_job_refs : list
        Jobs disponibles devueltos por `listar_jobs_si_corresponde`.

    Returns
    -------
    tuple
        (selected_job_ref, selected_job_name, selected_counts,
        selected_result_ids, sim_result, sim_counts, result_source,
        selected_results_df). En modo RUN NUEVO todos son None.

    Raises
    ------
    ValueError
        Si no hay ningun job disponible para seleccionar.
    IndexError
        Si el indice seleccionado esta fuera de rango.
    """
    if allow_new_execution:
        print("No aplica: ALLOW_NEW_EXECUTION = True (run nuevo completo); no se cargan resultados previos.")
        return None, None, None, None, None, None, None, None

    selected_job_index = job_selector.value
    if selected_job_index is None:
        raise ValueError("No hay jobs disponibles para seleccionar.")
    if not 0 <= selected_job_index < len(execution_job_refs):
        raise IndexError(f"Indice fuera de rango: {selected_job_index}")

    selected_job_ref = execution_job_refs[selected_job_index]
    selected_job_name, selected_job_status = resumir_job(selected_job_ref)

    print(f"Indice seleccionado: {selected_job_index}")
    print(f"Job: {selected_job_name}")
    print(f"Job ID: {selected_job_ref.id}")
    print(f"Estado: {selected_job_status}")

    selected_result_refs, selected_downloaded_results, selected_counts, selected_result_ids = (
        descargar_resultados_job(selected_job_ref)
    )
    selected_results_df = construir_tabla_resultados(selected_counts, selected_result_ids)

    sim_result = selected_downloaded_results[0]
    sim_counts = selected_counts[0] if len(selected_counts) == 1 else selected_counts

    return (
        selected_job_ref, selected_job_name, selected_counts, selected_result_ids,
        sim_result, sim_counts, "selected_nexus_job", selected_results_df,
    )


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


def compilar_si_corresponde(allow_new_execution, execution_target, circuito, suffix):
    """
    Compila y sube el HUGR a Nexus solo para un RUN NUEVO remoto
    (`execution_target="nexus_selene"`). En modo CONSULTA o RUN NUEVO
    local no compila nada.

    Parameters
    ----------
    allow_new_execution : bool
        Interruptor maestro del cuaderno (`ALLOW_NEW_EXECUTION`).
    execution_target : str
        Destino del run nuevo (`EXECUTION_TARGET`): "local" o
        "nexus_selene".
    circuito
        Funcion decorada con `@guppy`.
    suffix : str
        Sufijo unico para nombrar el HUGR subido.

    Returns
    -------
    tuple
        (hugr_binary, ref_hugr), ambos None si no aplica.

    Raises
    ------
    ValueError
        Si `execution_target` no es "local" ni "nexus_selene".
    """
    if not allow_new_execution:
        print("No aplica: modo CONSULTA (ALLOW_NEW_EXECUTION = False); no se compila ni sube HUGR.")
        return None, None

    if execution_target == "nexus_selene":
        hugr_binary, ref_hugr = compilar_y_subir_hugr(circuito, f"encoding-logical-plus-{suffix}")
        print("HUGR compilado y subido a Nexus:", ref_hugr)
        return hugr_binary, ref_hugr

    if execution_target == "local":
        print("Modo local: la compilacion ocurre dentro de la simulacion; no se sube HUGR a Nexus.")
        return None, None

    raise ValueError('EXECUTION_TARGET debe ser "local" o "nexus_selene"')


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


def enviar_job_selene(ref_hugr, n_qubits=2, n_shots=100, nombre=None, simulator="statevector"):
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
    simulator : str, optional
        "statevector" (por defecto) o "stabilizer". Mismo criterio que en
        `ejecutar_local`: statevector para rotaciones arbitrarias (kernel
        ZZ), stabilizer solo para programas Clifford.

    Returns
    -------
    job_ref
        Referencia al job de ejecucion enviado.

    Raises
    ------
    ValueError
        Si `simulator` no es "statevector" ni "stabilizer".
    """
    if simulator == "statevector":
        sim = qnx.models.StatevectorSimulator()
    elif simulator == "stabilizer":
        sim = qnx.models.StabilizerSimulator()
    else:
        raise ValueError('simulator debe ser "statevector" o "stabilizer"')

    sim_config = qnx.models.SeleneConfig(n_qubits=n_qubits, simulator=sim)
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


def ejecutar_si_corresponde(allow_new_execution, execution_target, circuito, ref_hugr, n_shots, suffix):
    """
    Ejecuta un RUN NUEVO segun el destino (local o remoto). En modo
    CONSULTA no ejecuta nada nuevo; se usan los resultados ya cargados.

    Parameters
    ----------
    allow_new_execution : bool
        Interruptor maestro del cuaderno (`ALLOW_NEW_EXECUTION`).
    execution_target : str
        Destino del run nuevo (`EXECUTION_TARGET`): "local" o
        "nexus_selene".
    circuito
        Funcion decorada con `@guppy`.
    ref_hugr
        Referencia al HUGR subido (solo requerida para destino remoto).
    n_shots : int
        Numero de repeticiones.
    suffix : str
        Sufijo unico para nombrar el job remoto.

    Returns
    -------
    tuple
        (local_result, local_counts, local_run_id, sim_job_ref,
        result_source). Los campos no aplicables quedan en None.

    Raises
    ------
    RuntimeError
        Si el destino es remoto y aun no se compilo/subio el HUGR.
    ValueError
        Si `execution_target` no es "local" ni "nexus_selene".
    """
    if not allow_new_execution:
        print("Modo CONSULTA: no se ejecuta simulacion nueva; se usan los resultados cargados arriba.")
        return None, None, None, None, None

    if execution_target == "local":
        local_result, local_counts = ejecutar_local(circuito, n_qubits=2, n_shots=n_shots, seed=42)
        local_run_id = nuevo_run_id_local()
        print("Simulacion local finalizada correctamente.")
        return local_result, local_counts, local_run_id, None, "local"

    if execution_target == "nexus_selene":
        if ref_hugr is None:
            raise RuntimeError("ref_hugr no existe; ejecuta primero la celda de compilacion/upload.")

        sim_job_ref = enviar_job_selene(ref_hugr, n_qubits=2, n_shots=n_shots, nombre=f"encoding-selene-sim-{suffix}")
        submit_status = qnx.jobs.status(sim_job_ref)

        print("Job enviado a Selene/Nexus correctamente.")
        print("sim_job_ref:", sim_job_ref)
        print("Estado inicial:", submit_status.status)
        print("Mensaje:", submit_status.message)
        print("La celda termina aqui; consulta el avance en la siguiente celda.")
        return None, None, None, sim_job_ref, None

    raise ValueError('EXECUTION_TARGET debe ser "local" o "nexus_selene"')


def consultar_job_remoto(sim_job_ref):
    """
    Consulta el estado de un job remoto nuevo sin bloquear el cuaderno.
    Si ya esta COMPLETED, descarga y tabula sus resultados.

    Parameters
    ----------
    sim_job_ref
        Referencia al job remoto enviado en la celda de ejecucion, o
        None si no se envio ninguno (local, modo CONSULTA, o aun no se
        ha ejecutado esa celda).

    Returns
    -------
    tuple
        (actualizado, sim_result, sim_counts, sim_result_ids,
        result_source). Si el job no esta COMPLETED, `actualizado` es
        False y los demas son None; el cuaderno debe conservar los
        valores previos en ese caso.

    Raises
    ------
    RuntimeError
        Si el job termino con error o fue cancelado.
    """
    if sim_job_ref is None:
        print("No hay sim_job_ref que consultar. Estas en modo local, en modo CONSULTA, o aun no enviaste un job remoto.")
        return False, None, None, None, None

    status = qnx.jobs.status(sim_job_ref)
    print("Estado:", status.status)
    print("Mensaje:", status.message)

    queue_position = getattr(status, "queue_position", None)
    if queue_position is not None:
        print("Posicion en cola:", queue_position)

    if "COMPLETED" in str(status.status):
        sim_result_refs, sim_downloaded_results, sim_counts_list, sim_result_ids = (
            descargar_resultados_job(sim_job_ref)
        )
        sim_result = sim_downloaded_results[0]
        sim_counts = sim_counts_list[0] if len(sim_counts_list) == 1 else sim_counts_list
        print("Job finalizado. Resultado descargado en sim_result; conteos en sim_counts.")
        return True, sim_result, sim_counts, sim_result_ids, "new_nexus_job"

    if "ERROR" in str(status.status) or "CANCELLED" in str(status.status):
        raise RuntimeError(f"El job termino sin exito: {status}")

    print("Job aun no finalizado. Vuelve a ejecutar esta celda mas tarde.")
    return False, None, None, None, None


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
    job_id, job_name). El CSV usa ";" como separador (no ",").

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
    df_run.to_csv(ruta_csv, index=False, sep=";")

    return ruta_csv


def guardar_resultado_actual(
    result_source,
    *,
    n_shots=None,
    local_counts=None,
    local_run_id=None,
    selected_counts=None,
    selected_job_ref=None,
    selected_job_name=None,
    selected_result_ids=None,
    sim_counts=None,
    sim_job_ref=None,
    sim_result_ids=None,
):
    """
    Despacha el guardado de la ejecucion cargada al caso que corresponda
    (simulador local, job de Nexus seleccionado o job de Nexus recien
    terminado) y devuelve la ruta del CSV generado.

    Envuelve la logica que antes vivia en la celda final del cuaderno:
    segun `result_source`, arma los argumentos del caso y llama a
    `guardar_ejecucion_csv`. Los tres casos producen el mismo formato de
    tabla.

    Parameters
    ----------
    result_source : str or None
        Origen de la ejecucion activa: "local", "selected_nexus_job" o
        "new_nexus_job".
    n_shots : int, optional
        Numero de shots (informativo).
    local_counts, local_run_id : optional
        Conteos e identificador de la ejecucion local.
    selected_counts, selected_job_ref, selected_job_name, selected_result_ids : optional
        Datos del job de Nexus seleccionado (modo consulta).
    sim_counts, sim_job_ref, sim_result_ids : optional
        Datos del job de Nexus remoto recien terminado.

    Returns
    -------
    str
        Ruta del archivo CSV generado.

    Raises
    ------
    RuntimeError
        Si no hay una ejecucion cargada, o si el caso local no tiene
        conteos.
    """
    if result_source == "local":
        if local_counts is None:
            raise RuntimeError("No hay resultados locales que guardar (local_counts esta vacio).")
        return guardar_ejecucion_csv(
            counts=local_counts,
            source="local",
            n_shots=n_shots,
            run_id=local_run_id,
        )

    if result_source == "selected_nexus_job":
        return guardar_ejecucion_csv(
            counts=selected_counts,
            source="selected_nexus_job",
            job_id=selected_job_ref.id,
            job_name=selected_job_name,
            result_ids=selected_result_ids,
            n_shots=n_shots,
        )

    if result_source == "new_nexus_job":
        return guardar_ejecucion_csv(
            counts=sim_counts,
            source="new_nexus_job",
            job_id=sim_job_ref.id,
            job_name=getattr(sim_job_ref.annotations, "name", None) or str(sim_job_ref.id),
            result_ids=sim_result_ids,
            n_shots=n_shots,
        )

    raise RuntimeError(
        "No hay una ejecucion cargada para guardar. Corre una simulacion local, "
        "selecciona un job de Nexus, o espera a que termine un job nuevo."
    )


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


def kernel_circuit_zz(x_i, x_j, remove_barriers=False):
    """
    Construye el circuito del kernel ``U(x_j)^dagger U(x_i)`` en pytket.

    La probabilidad de medir ``00...0`` en este circuito estima el kernel
    de fidelidad ``K(x_i, x_j) = |<phi(x_j)|phi(x_i)>|^2``.

    Parameters
    ----------
    x_i, x_j : array-like
        Vectores de features de las dos observaciones a comparar.
    remove_barriers : bool, optional
        Si True, elimina las barreras (necesario antes de cargar el
        circuito en guppy o enviarlo a un backend). Por defecto False,
        para conservarlas en la inspeccion visual.

    Returns
    -------
    pytket.Circuit
        Circuito del kernel para el par (x_i, x_j).
    """
    circuit_xi = zz_feature_map(x_i)
    circuit_xj_adjoint = zz_feature_map(x_j).dagger()

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
        (``zero_count / shots``).

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


def tabla_resumen_kernel(summary, row_i, row_j, source):
    """
    Crea la vista compacta (una fila) del resumen de un par del kernel.

    Parameters
    ----------
    summary : dict
        Resumen devuelto por `resumen_kernel_desde_resultado`.
    row_i, row_j : int
        Indices de las filas comparadas.
    source : str
        Origen de la ejecucion (por ejemplo "local_statevector").

    Returns
    -------
    pandas.DataFrame
        Tabla de una fila con el par y su resumen.
    """
    return pd.DataFrame([{
        "source": source,
        "row_i": int(row_i),
        "row_j": int(row_j),
        **summary,
    }])


def guardar_resumen_kernel_csv(
    summary,
    row_i,
    row_j,
    source,
    run_id=None,
    job_id=None,
    job_name=None,
    directory="data/runs",
):
    """
    Guarda en CSV el resumen compacto de una ejecucion de un par del
    kernel (una sola fila por ejecucion).

    Parameters
    ----------
    summary : dict
        Resumen devuelto por `resumen_kernel_desde_resultado`.
    row_i, row_j : int
        Indices de las filas comparadas.
    source : str
        Origen de la ejecucion.
    run_id : optional
        Identificador de la ejecucion; si no se pasa, se usa el job_id o
        se genera uno local nuevo.
    job_id, job_name : optional
        Identificadores del job de Nexus, si aplica.
    directory : str, optional
        Carpeta destino. Por defecto "data/runs".

    Returns
    -------
    pathlib.Path
        Ruta del CSV generado (``kernel_run_<run_id>.csv``).
    """
    if run_id is None:
        run_id = str(job_id) if job_id is not None else f"local-{uuid.uuid4().hex[:12]}"

    row = {
        "run_id": str(run_id),
        "source": source,
        "job_id": "" if job_id is None else str(job_id),
        "job_name": "" if job_name is None else str(job_name),
        "row_i": int(row_i),
        "row_j": int(row_j),
        "n_qubits": len(summary["zero_state"]),
        "zero_state": summary["zero_state"],
        "zero_count": summary["zero_count"],
        "shots": summary["shots"],
        "kernel_rate": summary["kernel_rate"],
    }

    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_run_id = str(run_id).replace("/", "_").replace("\\", "_")
    output_path = output_dir / f"kernel_run_{safe_run_id}.csv"
    pd.DataFrame([row]).to_csv(output_path, index=False)
    return output_path


def crear_programa_kernel_guppy(x_i, x_j):
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

    Returns
    -------
    tuple
        (pair_program, pair_circuit): el programa guppy verificado con
        ``.check()`` y el circuito pytket subyacente.
    """
    pair_circuit = kernel_circuit_zz(
        x_i,
        x_j,
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

    Returns
    -------
    dict
        Con claves: kernel (la tasa K(i,j)), summary, counts, program,
        pytket_circuit y raw_result.
    """
    pair_program, pair_circuit = crear_programa_kernel_guppy(x_i, x_j)
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
):
    """
    Guarda en CSV el resumen compacto de una construccion de matriz
    kernel (una fila por circuito ejecutado).

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

    Returns
    -------
    pathlib.Path
        Ruta del CSV generado (``kernel_matrix_run_<run_id>.csv``).
    """
    if run_id is None:
        run_id = f"matrix-local-{uuid.uuid4().hex[:12]}"

    run_df = matrix_result["run_summary"].copy()
    run_df.insert(0, "job_id", "" if job_id is None else str(job_id))
    run_df.insert(0, "source", source)
    run_df.insert(0, "run_id", str(run_id))

    output_dir = Path("data/runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_run_id = str(run_id).replace("/", "_").replace("\\", "_")
    output_path = output_dir / f"kernel_matrix_run_{safe_run_id}.csv"
    run_df.to_csv(output_path, index=False)
    return output_path
