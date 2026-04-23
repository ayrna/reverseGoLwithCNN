import numpy as np

def generador(n_tableros, dim_tablero, densidad_min, densidad_max, batch_size=5000):
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

