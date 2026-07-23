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
from tqdm import tqdm


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


def kernel_circuit_zz(x_i, x_j, remove_barriers=False):
    """
    Construye el circuito del kernel ZZ ``U(x_j)^dagger U(x_i)``.

    Wrapper retrocompatible de `kernel_circuit` con ``feature_map="zz"``.

    Parameters
    ----------
    x_i, x_j : array-like
        Vectores de features de las dos observaciones a comparar.
    remove_barriers : bool, optional
        Si True, elimina las barreras. Por defecto False.

    Returns
    -------
    pytket.Circuit
        Circuito del kernel ZZ para el par (x_i, x_j).
    """
    return kernel_circuit(
        x_i, x_j, feature_map="zz", remove_barriers=remove_barriers
    )


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


def cargar_datos_kernel(ruta="data/processed/df_escalado.csv"):
    """
    Carga el dataset escalado del kernel y lo separa en train/test segun
    la columna ``_PartInd_`` (0 = train, 1 = test).

    Parameters
    ----------
    ruta : str, optional
        Ruta del CSV escalado (separador ";"). Por defecto
        "data/processed/df_escalado.csv".

    Returns
    -------
    tuple
        (kernel_df, feature_columns, train_df, test_df): el dataset
        completo, la lista de columnas de features (excluye Potability y
        _PartInd_) y las particiones train/test solo con features.
    """
    kernel_df = pd.read_csv(ruta, sep=";")
    feature_columns = [
        column
        for column in kernel_df.columns
        if column not in {"Potability", "_PartInd_"}
    ]

    train_df = (
        kernel_df.loc[kernel_df["_PartInd_"] == 0, feature_columns]
        .reset_index(drop=True)
    )
    test_df = (
        kernel_df.loc[kernel_df["_PartInd_"] == 1, feature_columns]
        .reset_index(drop=True)
    )
    return kernel_df, feature_columns, train_df, test_df


NIVELES_MUESTREO = {"3": 3, "3+2": 2, "3+2+1": 1}


def cargar_datos_kernel_muestreo(
    ruta_excel,
    nivel_train,
    nivel_test,
    sheet_name="muestreos",
    objetivo="Potability",
    particion="_PartInd_",
    columna_muestreo="_Muestreo_",
):
    """
    Carga el subconjunto de filas de la hoja "muestreos" (generada por
    `asignar_muestreo_balanceado` + `guardar_muestreo_excel` en el pipeline
    v1) y lo separa en train/test, filtrando cada particion por su propio
    nivel de muestreo jerarquico.

    Por el anidamiento del muestreo, la submuestra de nivel ``k`` son las
    filas con ``_Muestreo_ >= k`` (ver docstring de
    `asignar_muestreo_balanceado`): nivel 3 es la mas profunda (mas
    pequena), nivel 1 es la muestra completa. El nombre de los niveles en
    `NIVELES_MUESTREO` seguido la notacion del usuario: "3" (solo la
    submuestra mas chica), "3+2" (union de las dos mas chicas) y "3+2+1"
    (la muestra completa).

    Nota importante: la hoja "muestreos" viene del pipeline v1 (entorno
    qsvm), con su propio split e imputacion/escalado. No esta alineada
    fila a fila con `data/processed/df_escalado.csv` (pipeline del
    handoff/guppy): son dos escalados independientes del mismo dataset
    crudo, asi que no se deben cruzar por indice. Usar esta funcion
    implica que el kernel consume las features escaladas del pipeline v1
    para las filas seleccionadas, no las de `df_escalado.csv`.

    Parameters
    ----------
    ruta_excel : str
        Ruta del Excel con la hoja de muestreos (por ejemplo
        "data/processed/dataset_v1.xlsx").
    nivel_train, nivel_test : int
        Nivel minimo de muestreo (1, 2 o 3) para train y test,
        respectivamente. Independientes entre si. Usar
        `NIVELES_MUESTREO["3"|"3+2"|"3+2+1"]` para traducir la notacion
        del usuario a este entero.
    sheet_name : str, optional
        Hoja del Excel. Por defecto "muestreos".
    objetivo : str, optional
        Columna objetivo a excluir de las features. Por defecto
        "Potability".
    particion : str, optional
        Columna de particion train/test. Por defecto "_PartInd_".
    columna_muestreo : str, optional
        Columna con el nivel de muestreo. Por defecto "_Muestreo_".

    Returns
    -------
    tuple
        (kernel_df, feature_columns, train_df, test_df): el subconjunto
        completo de la hoja (sin filtrar por nivel), la lista de columnas
        de features y las particiones train/test ya filtradas por su
        nivel, solo con features.

    Raises
    ------
    ValueError
        Si algun nivel no esta en {1, 2, 3} o el filtro resultante queda
        vacio para alguna particion.
    """
    if nivel_train not in (1, 2, 3):
        raise ValueError("nivel_train debe ser 1, 2 o 3.")
    if nivel_test not in (1, 2, 3):
        raise ValueError("nivel_test debe ser 1, 2 o 3.")

    kernel_df = pd.read_excel(ruta_excel, sheet_name=sheet_name)
    feature_columns = [
        column
        for column in kernel_df.columns
        if column not in {objetivo, particion, columna_muestreo}
    ]

    train_df = (
        kernel_df.loc[
            (kernel_df[particion] == 0) & (kernel_df[columna_muestreo] >= nivel_train),
            feature_columns,
        ]
        .reset_index(drop=True)
    )
    test_df = (
        kernel_df.loc[
            (kernel_df[particion] == 1) & (kernel_df[columna_muestreo] >= nivel_test),
            feature_columns,
        ]
        .reset_index(drop=True)
    )

    if train_df.empty:
        raise ValueError(f"El filtro de train (nivel {nivel_train}) no devolvio filas.")
    if test_df.empty:
        raise ValueError(f"El filtro de test (nivel {nivel_test}) no devolvio filas.")

    return kernel_df, feature_columns, train_df, test_df


def seleccionar_par_kernel(train_df, row_i, row_j, feature_map="zz"):
    """
    Valida los indices de un par, construye su circuito de inspeccion
    (con barreras, para visualizarlo) e imprime sus dimensiones.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Particion de train con solo features.
    row_i, row_j : int
        Indices de las filas a comparar.
    feature_map : str or callable, optional
        Feature map a usar ("zz", "zyy" o "ry_cx_rx"). Por defecto "zz".

    Returns
    -------
    tuple
        (x_i, x_j, preview_circuit): los vectores de features y el
        circuito pytket del kernel con barreras.

    Raises
    ------
    IndexError
        Si algun indice es negativo o queda fuera de train.
    """
    if min(row_i, row_j) < 0:
        raise IndexError("Los indices no pueden ser negativos.")
    if max(row_i, row_j) >= len(train_df):
        raise IndexError("row_i o row_j queda fuera de train.")

    x_i = train_df.iloc[row_i].to_numpy(dtype=float)
    x_j = train_df.iloc[row_j].to_numpy(dtype=float)
    preview_circuit = kernel_circuit(
        x_i, x_j, feature_map=feature_map, remove_barriers=False
    )

    print(f"Kernel seleccionado: train[{row_i}] vs train[{row_j}]")
    print("Qubits:", preview_circuit.n_qubits)
    print("Puertas:", preview_circuit.n_gates)
    print("Profundidad:", preview_circuit.depth())
    return x_i, x_j, preview_circuit


MATRIX_BACKEND_OPTIONS = [
    "local_selene_statevector",
    "nexus_selene_statevector",
    "H1-1LE",
    "H1-Emulator",
    "H2-1LE",
    "H2-Emulator",
    "Helios-1E-lite",
]


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
