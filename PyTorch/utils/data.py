import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from utils.gol import GameOfLife

def generator(n_tableros, dim_tablero, densidad_min, densidad_max, batch_size=5000):
    """
    Generator of Conway's Game of Life base states.

    Arguments:
    
    n_tableros (int): Number of states to be generated. 
    dim_tablero (tuple): State shape.
    densidad_min (float): Minimum density of the generated boards.
    densidad_max (float): Maximum density of the generated boards.
    batch_size (int): number of states to be simultaneously generated.

    Outputs:
    
    tableros_finales (array): contains the base boards
    
    """

    # --------- Paso 1: Datos del tablero ---------
    H, W = dim_tablero
    num_celulas = H * W
    
    # --------- Paso 2: Cálculo de números 1 mínimo y máximo ---------
    min_celulas = int(np.ceil(densidad_min * num_celulas))
    
    max_celulas = int(np.floor(densidad_max * num_celulas))
    
    # Generamos una distribución lineal de unos
    conteos_enteros = np.linspace(min_celulas, max_celulas, n_tableros) # Array de n_tableros floats
    conteos_exactos = np.round(conteos_enteros).astype(int) 
    np.random.shuffle(conteos_exactos)
    
    total_batches = int(np.ceil(n_tableros / batch_size)) 
    
    for i in range(total_batches):

        # Índices de la creación de tableros
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_tableros)
        current_batch_size = end_idx - start_idx
        
        # Obtenemos los '1's que tocan para este lote
        batch_counts = conteos_exactos[start_idx:end_idx]
        
        # --- Lógica de generación exacta  ---
        
        # 1. Matriz de índices base
        indices = np.arange(num_celulas)[np.newaxis, :] 
        
        # 2. Poner los unos en orden (izq -> der) [11111111111110000000000...]
        tableros_ordenados = (indices < batch_counts[:, np.newaxis]).astype(np.int8)
        
        # 3. Generar permutación aleatoria de las células
        permutacion = np.argsort(np.random.random((current_batch_size, num_celulas)), axis=1)
        
        # 4. Mezclar los 1
        tableros_flat = np.take_along_axis(tableros_ordenados, permutacion, axis=1)
        tableros_finales = tableros_flat.reshape(current_batch_size, H, W)
        
        yield tableros_finales

def waterfilling(densidades, n_objetivo=200000, n_bins=150, rango=(0.05, 0.40)):
    """
    Selecciona exactamente n_objetivo muestras intentando maximizar la uniformidad.
    Usa estrategia de 'Water Filling' para compensar bins vacíos con bins llenos.
    """
    
    # 1. Identificar a qué bin pertenece cada tablero
    bins = np.linspace(rango[0], rango[1], n_bins + 1) 
    indices_bins = np.digitize(densidades, bins) - 1
    indices_bins = np.clip(indices_bins, 0, n_bins - 1)
    
    # 2. Contar disponibilidad real por bin
    # conteos_disponibles[i] = cuántos tableros existen realmente en el bin i
    conteos_disponibles = np.bincount(indices_bins, minlength=n_bins)
    
    # 3. Encontrar el "Nivel de Corte" óptimo (Búsqueda Binaria)
    # Buscamos un número L tal que si tomamos min(disponible, L) de cada bin,
    # la suma total se acerque lo más posible a n_objetivo sin pasarse.
    low = 0
    high = np.max(conteos_disponibles) # El bin más alto
    nivel_base = 0
    
    while low <= high:
        mid = (low + high) // 2
        # Cuántos tomaríamos si el límite fuera 'mid'
        suma_estimada = np.sum(np.minimum(conteos_disponibles, mid))
        
        if suma_estimada <= n_objetivo:
            nivel_base = mid
            low = mid + 1
        else:
            high = mid - 1
            
    # 4. Calcular asignación base y el faltante
    # asignacion_final[i] = cuántos vamos a tomar del bin i
    asignacion_final = np.minimum(conteos_disponibles, nivel_base)
    faltante = n_objetivo - np.sum(asignacion_final)
    
    # 5. Repartir el remanente (faltante)
    # El 'faltante' será pequeño (menor que n_bins). Lo repartimos uno a uno
    # entre los bins que todavía tienen tableros de sobra (donde disponible > nivel_base).
    indices_con_superavit = np.where(conteos_disponibles > nivel_base)[0]
    
    if faltante > 0 and len(indices_con_superavit) > 0:
        # Elegimos al azar quiénes ponen el extra para no sesgar
        bins_extra = np.random.choice(indices_con_superavit, size=int(faltante), replace=False)
        for bin_idx in bins_extra:
            asignacion_final[bin_idx] += 1
            
    # 6. Ejecutar la selección
    indices_seleccionados = []
    for b in range(n_bins):
        n_tomar = asignacion_final[b]
        if n_tomar > 0:
            # Índices originales que caen en este bin
            idx_en_bin = np.where(indices_bins == b)[0]
            # Muestreo aleatorio sin reemplazo
            seleccion = np.random.choice(idx_en_bin, size=n_tomar, replace=False)
            indices_seleccionados.append(seleccion)
            
    # Unir y mezclar
    indices = np.concatenate(indices_seleccionados)
    np.random.shuffle(indices)
    
    return indices

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

def data_generator(seed, delta, shape, dir_npz, name, 
                    TrainVal=True, val_split=0.2, n_objetivo=60_000, 
                    num_tableros = 120_000, densidad_min= 0.05, densidad_max = 0.95, batch_size=5_000):
    """
    Wrapper for the generation of Conway's Game of Life states that fulfill certain conditions. This function returns a 
    npz file with the original (base) state, the initial state obtained after 3 warmup generations and its corresponding 
    final state after delta generations.

    Arguments:
    
    seed (int): random number seed. 
    delta (int): number of generations between initial and final state.
    shape (tuple): size of the states.
    dir_npz (string): Parent directory in which npz files must be saved.
    name (string): Name of the npz file.
    TrainVal (bool): Paramater to distinguish between Train&Val (True) and Test (False) phases. 
    val_split (float): Percentage for spliting the training dataset in train and validation datasets.
    n_objetivo (int): Number of states of the training dataset before split.
    num_tableros (int): Number of states to be generated. It must be larger than n_objetivo to ensure that the whole density range is covered.
    densidad_min (float): Minimum density of the generated boards.
    densidad_max (float): Maximum density of the generated boards.
    batch_size (int): number of states to be simultaneously generated.

    Outputs:
    
    Save the npz files with Conway's Game of Life states in the specified path.

    """
    # ---------- Paso 1: generador de tableros ------------
    generador = generator(num_tableros, 
                                  shape, 
                                  densidad_min, 
                                  densidad_max, 
                                  batch_size) # (k, H, W)

    bases = []
    for tableros in generador: 
        bases.append(tableros)

    bases = np.concatenate(bases, axis=0) # (k, H, W)

    # ---------- Paso 2: Jugamos a Conway ------------
    iniciales= []
    finales = []
    gol3 = GameOfLife(3)
    gol5 = GameOfLife(delta)
    for tablero in bases:
        # Tableros iniciales: 3 generaciones de calentamiento
        tablero_inicial = gol3.evolution(tablero, 'torch').squeeze() 
        iniciales.append(tablero_inicial) 

        # Tableros finales
        tablero_final = gol5.evolution(tablero_inicial, 'torch').squeeze()
        finales.append(tablero_final)
    
    iniciales = np.array(iniciales) # (k, H, W)
    finales = np.array(finales) # (k, H, W)

    # --------- Paso 3: Limpieza de tableros según tableros finales -----------
    densidades_finales = np.mean(finales, axis=(1,2))
    mask = (densidades_finales > densidad_min) & (densidades_finales < densidad_max)

    bases = bases[mask] # (k, H, W)
    iniciales = iniciales[mask] # (k, H, W)
    finales= finales[mask] # (k, H, W)

    # --------- Paso 4: Filtro de únicos and reshape --------------
    _, indices = np.unique(finales.astype(np.int8), axis=0, return_index=True)

    bases = bases[indices] # (k, H, W)
    iniciales = iniciales[indices] # (k, H, W)
    finales = finales[indices] # (k, H, W)

    bases = bases.reshape(-1, shape[0]*shape[1]) # (k, H*W)
    iniciales = iniciales.reshape(-1, shape[0]*shape[1])  # (k, H*W)
    finales = finales.reshape(-1, shape[0]*shape[1]) # (k, H*W)

    # --------- Paso 5: Aplicamos Water Filling -----------
    # Selecciono 50k tableros
    densidades = np.mean(finales,axis=1)
    max = np.max(densidades)
    indices = waterfilling(densidades, n_objetivo=n_objetivo, n_bins=300, rango=(0.05, max + 0.05))
     
    bases = bases[indices] # (k, H*W)
    iniciales = iniciales[indices] # (k, H*W)
    finales = finales[indices] # (k, H*W)

    del densidades, max, mask, indices

    if TrainVal:
        # ---------- Paso 6: División de dataset Train y Val -----------
        indices = np.arange(len(bases))
        rng = np.random.default_rng(seed) # Generador aislado
        rng.shuffle(indices)

        bases = bases[indices]
        iniciales = iniciales[indices]
        finales = finales[indices]

        num_val = int(len(bases) * (1 - val_split))
        bases_train, iniciales_train, finales_train = bases[:num_val], iniciales[:num_val], finales[:num_val]
        bases_val, iniciales_val, finales_val = bases[num_val:], iniciales[num_val:], finales[num_val:]

        # ---------- Paso 7: Creación de npz -----------
        create_npz(delta, bases_train, iniciales_train, finales_train, dir_npz[0], name[0]) # (k, H*W), (k, H*W), (k, H*W)
        create_npz(delta, bases_val, iniciales_val, finales_val, dir_npz[1], name[1])
    else:
        create_npz(delta, bases, iniciales, finales, dir_npz, name) 

#
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
