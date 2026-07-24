"""Validación reproducible de los artefactos versionados del proyecto.

Uso:
    python validar_repositorio.py

El script no modifica archivos. Termina con código distinto de cero si detecta
una inconsistencia en el dataset procesado, el muestreo o las matrices kernel.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "water_potability.csv"
PROCESSED = ROOT / "data" / "processed" / "dataset_v1.xlsx"
RUNS = ROOT / "data" / "runs"
FEATURES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]
TARGET = "Potability"
EXPECTED_SHEETS = [
    "originales",
    "imputados",
    "normalizados",
    "muestreos",
    "variables",
]
MATRIX_FAMILIES = {
    "ZZ": "zz",
    "zyy": "zyy",
    "Custom": "ry_cx_rx",
}
LEVEL_BY_SIZE = {16: 3, 32: 2}


def require(condition, message):
    """Lanza una excepción con un mensaje legible si una regla no se cumple."""
    if not condition:
        raise AssertionError(message)


def sample_rows(samples, partition, level, columns):
    mask = (
        (samples["_PartInd_"] == partition)
        & (samples["_Muestreo_"] >= level)
    )
    return samples.loc[mask, columns].reset_index(drop=True)


def nested_positions(subset, superset, variables):
    """Ubica cada fila de una muestra anidada dentro de la muestra mayor."""
    positions = []
    for _, row in subset.iterrows():
        equal = np.isclose(
            superset[variables].to_numpy(dtype=float),
            row[variables].to_numpy(dtype=float),
            rtol=0,
            atol=1e-12,
        ).all(axis=1)
        equal &= superset[TARGET].to_numpy() == row[TARGET]
        found = np.flatnonzero(equal)
        require(
            len(found) == 1,
            "Cada observación anidada debe tener una coincidencia única.",
        )
        positions.append(int(found[0]))
    return positions


def validate_dataset():
    require(RAW.is_file(), f"Falta el dataset original: {RAW}")
    require(PROCESSED.is_file(), f"Falta el dataset procesado: {PROCESSED}")

    raw = pd.read_csv(RAW)
    require(raw.shape == (3276, 10), f"Forma inesperada del CSV original: {raw.shape}")
    require(raw.columns.tolist() == [*FEATURES, TARGET], "Columnas originales inesperadas.")
    require(set(raw[TARGET].unique()) == {0, 1}, "Potability debe ser binaria (0/1).")

    workbook = pd.ExcelFile(PROCESSED)
    require(workbook.sheet_names == EXPECTED_SHEETS, "Las hojas de dataset_v1.xlsx cambiaron.")
    sheets = {
        name: pd.read_excel(workbook, sheet_name=name)
        for name in EXPECTED_SHEETS
    }

    for name in ("originales", "imputados", "normalizados"):
        require(
            sheets[name].shape[0] == len(raw),
            f"La hoja {name} no conserva las {len(raw)} observaciones.",
        )

    sort_columns = [TARGET, *FEATURES]
    raw_sorted = raw.sort_values(sort_columns, na_position="first").reset_index(drop=True)
    original_sorted = (
        sheets["originales"][[*FEATURES, TARGET]]
        .sort_values(sort_columns, na_position="first")
        .reset_index(drop=True)
    )
    require(
        np.allclose(
            raw_sorted[FEATURES].to_numpy(dtype=float),
            original_sorted[FEATURES].to_numpy(dtype=float),
            rtol=0,
            atol=1e-8,
            equal_nan=True,
        )
        and raw_sorted[TARGET].equals(original_sorted[TARGET]),
        "La hoja originales no reproduce el contenido del CSV de entrada.",
    )

    require(
        not sheets["imputados"][FEATURES].isna().any().any(),
        "La hoja imputados todavía contiene valores faltantes.",
    )
    require(
        not sheets["normalizados"][FEATURES].isna().any().any(),
        "La hoja normalizados contiene valores faltantes.",
    )

    normalized_train = sheets["normalizados"].loc[
        sheets["normalizados"]["_PartInd_"] == 0, FEATURES
    ]
    require(
        np.allclose(normalized_train.mean().to_numpy(), 0, atol=1e-12),
        "La media de train normalizado no es cero.",
    )
    require(
        np.allclose(normalized_train.std(ddof=0).to_numpy(), 1, atol=1e-12),
        "La desviación estándar poblacional de train normalizado no es uno.",
    )

    variables = (
        sheets["variables"].iloc[:, 0].dropna().astype(str).tolist()
    )
    require(variables, "La hoja variables está vacía.")
    require(
        len(variables) == len(set(variables)) and set(variables) <= set(FEATURES),
        "La hoja variables contiene nombres duplicados o desconocidos.",
    )

    samples = sheets["muestreos"]
    normalized = sheets["normalizados"]
    normalized_values = normalized[FEATURES].to_numpy(dtype=float)
    normalized_target = normalized[TARGET].to_numpy()
    normalized_partition = normalized["_PartInd_"].to_numpy()
    for _, row in samples.iterrows():
        match = np.isclose(
            normalized_values,
            row[FEATURES].to_numpy(dtype=float),
            rtol=0,
            atol=1e-12,
        ).all(axis=1)
        match &= normalized_target == row[TARGET]
        match &= normalized_partition == row["_PartInd_"]
        require(
            match.any(),
            "Una fila de muestreos no pertenece al dataset normalizado correspondiente.",
        )

    expected_counts = {
        (0, 1): 64,
        (1, 1): 32,
        (0, 2): 32,
        (1, 2): 16,
        (0, 3): 16,
        (1, 3): 8,
    }
    for (partition, level), expected in expected_counts.items():
        selected = samples.loc[
            (samples["_PartInd_"] == partition)
            & (samples["_Muestreo_"] >= level)
        ]
        require(
            len(selected) == expected,
            f"Muestreo partición={partition}, nivel>={level}: "
            f"{len(selected)} filas; se esperaban {expected}.",
        )
        counts = selected[TARGET].value_counts()
        require(
            len(counts) == 2 and counts.nunique() == 1,
            f"El muestreo partición={partition}, nivel>={level} no está balanceado.",
        )

    return sheets, variables


def validate_matrices(samples, variables, nested_tolerance=0.09):
    identity_columns = [*variables, TARGET]
    train16 = sample_rows(samples, 0, LEVEL_BY_SIZE[16], identity_columns)
    train32 = sample_rows(samples, 0, LEVEL_BY_SIZE[32], identity_columns)
    test8 = sample_rows(samples, 1, LEVEL_BY_SIZE[16], identity_columns)
    test16 = sample_rows(samples, 1, LEVEL_BY_SIZE[32], identity_columns)
    train_positions = nested_positions(train16, train32, variables)
    test_positions = nested_positions(test8, test16, variables)

    matrices = {}
    report = []
    for filename_prefix, family_name in MATRIX_FAMILIES.items():
        for size in (16, 32):
            train_path = RUNS / f"{filename_prefix}_Train_{size}.csv"
            test_path = RUNS / f"{filename_prefix}_Test_{size}.csv"
            require(train_path.is_file(), f"Falta {train_path}")
            require(test_path.is_file(), f"Falta {test_path}")

            K_train = pd.read_csv(train_path, sep=";", index_col=0).to_numpy(float)
            K_test = pd.read_csv(test_path, sep=";", index_col=0).to_numpy(float)
            n_test = size // 2

            require(K_train.shape == (size, size), f"Forma inválida en {train_path.name}.")
            require(K_test.shape == (n_test, size), f"Forma inválida en {test_path.name}.")
            require(
                np.isfinite(K_train).all() and np.isfinite(K_test).all(),
                f"{family_name}_{size} contiene NaN o infinitos.",
            )
            require(
                K_train.min() >= -1e-12
                and K_test.min() >= -1e-12
                and K_train.max() <= 1 + 1e-12
                and K_test.max() <= 1 + 1e-12,
                f"{family_name}_{size} contiene valores fuera de [0, 1].",
            )
            require(
                np.allclose(K_train, K_train.T, atol=1e-12, rtol=0),
                f"{family_name}_{size} no es simétrica.",
            )
            require(
                np.allclose(np.diag(K_train), 1, atol=1e-12, rtol=0),
                f"{family_name}_{size} no tiene diagonal unitaria.",
            )

            min_eigenvalue = float(np.linalg.eigvalsh(K_train).min())
            require(
                min_eigenvalue >= -0.1,
                f"{family_name}_{size} es fuertemente indefinida "
                f"(autovalor mínimo={min_eigenvalue:.6f}).",
            )
            matrices[(filename_prefix, size)] = (K_train, K_test)
            report.append(
                {
                    "familia": family_name,
                    "train": size,
                    "test": n_test,
                    "autovalor_min": min_eigenvalue,
                }
            )

        train_small, test_small = matrices[(filename_prefix, 16)]
        train_large, test_large = matrices[(filename_prefix, 32)]
        train_delta = float(
            np.abs(
                train_small
                - train_large[np.ix_(train_positions, train_positions)]
            ).max()
        )
        test_delta = float(
            np.abs(
                test_small
                - test_large[np.ix_(test_positions, train_positions)]
            ).max()
        )
        require(
            max(train_delta, test_delta) <= nested_tolerance,
            f"{family_name}: las matrices 16/32 no son coherentes con las muestras "
            f"anidadas (diferencia máxima={max(train_delta, test_delta):.6f}).",
        )
        for row in report:
            if row["familia"] == family_name:
                row["delta_anidado_max"] = max(train_delta, test_delta)

    return pd.DataFrame(report)


def main():
    sheets, variables = validate_dataset()
    report = validate_matrices(sheets["muestreos"], variables)
    print("VALIDACIÓN COMPLETA: OK")
    print(f"Variables seleccionadas ({len(variables)}): {', '.join(variables)}")
    print(report.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
