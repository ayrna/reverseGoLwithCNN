import os
import numpy as np
from pathlib import Path
from scipy import sparse

def load_npz(ruta_archivo, nombre_archivo, log=False):
    
    """
    Loads the Conway's Game of Life states from a npz file.

    Arguments:
    
    ruta_archivo (string): path to the npz file.
    nombre_archivo (string): name of the npz file.
    log (Bool): set to True to receive logs in terminal.
    
    Outputs:
    
    tableros_base (array): contains the base states.
    tableros_iniciales (array): contains the initial states.
    tableros_finales (array): contains the final states.

    """
    # Inicializamos listas vacías para tableros.
    sparse_base = []
    sparse_iniciales = []
    sparse_finales = []

    try:
        # Carga el archivo .npz
        npz_file = np.load(ruta_archivo, allow_pickle=True)
        
        # Reconstruyo la matriz dispersa (CSR) a partir de los datos guardados
        X = sparse.csr_matrix(
            (npz_file['data'], npz_file['indices'], npz_file['indptr']),
            shape=npz_file['shape']
        )
        M = npz_file['M'].item()

        # Almacena la matriz dispersa y los metadatos en un diccionario
        if log:
            print(f"Archivo {nombre_archivo} cargado exitosamente.")
        
        # Añado a la lista.
        sparse_base.append(X[:,0:M])
        sparse_iniciales.append(X[:,M:2*M])
        sparse_finales.append(X[:, 2*M:3*M])
        
    except Exception as e:
        print(f"Error al cargar el archivo {nombre_archivo}: {e}")

    if sparse_base:
        # print("\nTableros Base cargados y listos para usar")

        # Combino todos los tableros en una única matriz sparse:
        sparse_tableros_base = sparse.vstack(sparse_base) 
        # Convierto los tableros a np.array()
        matriz_densa_base = sparse_tableros_base.toarray() 
        # Convierto matrices a binarias.
        tableros_base = (matriz_densa_base > 0).astype(int)
        # print(tableros_base.shape)
    else:
        print("No se encontraron archivos .npz para cargar.")

    if sparse_iniciales:
        # print("\nTableros Iniciales cargados y listos para usar")

        # Combino todos los tableros en una única matriz sparse:
        sparse_tableros_iniciales = sparse.vstack(sparse_iniciales) 
        # Convierto los tableros a np.array()
        matriz_densa_init = sparse_tableros_iniciales.toarray() 
        # Convierto matrices a binarias.
        tableros_iniciales = (matriz_densa_init > 0).astype(int)
        # print(tableros_iniciales.shape)
    else:
        print("No se encontraron archivos .npz para cargar.")

    if sparse_finales:
        # print("\nTableros finales cargados y listos para usar")

        # Combino todos los tableros en una única matriz sparse:
        sparse_tableros_finales = sparse.vstack(sparse_finales) 
        # Convierto los tableros a np.array()
        matriz_densa_fin = sparse_tableros_finales.toarray() 
        # Convierto matrices a binarias.
        tableros_finales = (matriz_densa_fin > 0).astype(int)
        # print(tableros_finales.shape)
    else:
        print("No se encontraron archivos .npz para cargar.")

    return tableros_base, tableros_iniciales, tableros_finales  # (k, H*W)


        
    