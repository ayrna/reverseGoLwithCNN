import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse

def create_npz(delta, tableros_base, tableros_iniciales, tableros_finales, dir_npz, name):
    """
    Creates a npz file with the base, initial and final Conway's Game of Life states.

    Arguments:
    
    delta (int): number of generations between initial and final state.
    tableros_base (array): base states.
    tableros_iniciales (array): initial states.
    tableros_finales (array): final states.
    dir_npz (string): Parent directory in which npz files must be saved.
    name (string): Name of the npz file.
 
    Outputs:
    
    npz files with Conway's Game of Life states. 

    """

    # Listas simples para id y delta
    ids, deltas = [], []
    bases, iniciales, finales = [], [], []

    id_tablero = 1
    for tablero_base, tablero_inicial, tablero_final in zip(tableros_base, tableros_iniciales, tableros_finales):
        ids.append(id_tablero)
        deltas.append(delta)
        bases.append(np.ravel(np.asarray(tablero_base,   dtype=np.bool_), order='C'))
        iniciales.append(np.ravel(np.asarray(tablero_inicial, dtype=np.bool_), order='C'))
        finales.append(np.ravel(np.asarray(tablero_final,  dtype=np.bool_), order='C'))
        id_tablero += 1

    # Convertimos a arrays 2D (N, M) donde M = nº de celdas (ej: 400)
    bases = np.stack(bases)
    iniciales = np.stack(iniciales)
    finales = np.stack(finales)

    M = bases.shape[1]   # número de celdas del tablero (ej. 400)

    # Crear los nombres de columnas
    # cols_id_delta = ["id", "delta"]
    cols_base     = [f"base_{i+1}"  for i in range(M)]
    cols_start    = [f"start_{i+1}" for i in range(M)]
    cols_stop     = [f"stop_{i+1}"  for i in range(M)]

    # Construimos DataFrames con SparseDtype para ahorrar memoria
    df_id_delta = pd.DataFrame({"id": ids, "delta": deltas})

    df_bases     = pd.DataFrame(bases, columns=cols_base).astype(pd.SparseDtype("bool", False))
    df_iniciales = pd.DataFrame(iniciales, columns=cols_start).astype(pd.SparseDtype("bool", False))
    df_finales   = pd.DataFrame(finales, columns=cols_stop).astype(pd.SparseDtype("bool", False))

    # Concatenamos todo en un único DataFrame
    df_full = pd.concat([df_id_delta, df_bases, df_iniciales, df_finales], axis=1)

    # Metadatos (ids y deltas)
    ids_np    = df_full["id"].to_numpy(dtype=np.uint32, copy=False)
    deltas_np = df_full["delta"].to_numpy(dtype=np.uint8,  copy=False)

    # DataFrames sparse -> matrices dispersas (sin densificar)
    A_base  = df_full[cols_base].sparse.to_coo().tocsr().astype(np.bool_, copy=False)
    A_start = df_full[cols_start].sparse.to_coo().tocsr().astype(np.bool_, copy=False)
    A_stop  = df_full[cols_stop].sparse.to_coo().tocsr().astype(np.bool_, copy=False)

    # Concatena por columnas: [base | start | stop] => X (N, 3M)
    X = sparse.hstack([A_base, A_start, A_stop], format="csr", dtype=np.bool_)
    
    out_path = Path(dir_npz)/f'{name}.npz'
    # Guarda componentes CSR + meta en un único .npz
    np.savez_compressed(
        out_path,
        data=X.data,
        indices=X.indices,
        indptr=X.indptr,
        shape=np.array(X.shape, dtype=np.int64),
        ids=ids_np,
        deltas=deltas_np,
        M=np.array(M, dtype=np.int64)
    )
    