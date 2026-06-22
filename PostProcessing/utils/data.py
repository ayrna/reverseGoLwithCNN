import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from utils.gol import GameOfLife
from torchmetrics.functional import roc, auroc, precision_recall_curve
from sklearn.model_selection import train_test_split
from utils.models import DiffGoLModel

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

def create_train_dataset(path2data:Path, path2save:Path, model_name:str='DiffGoL', shape:tuple=(15,15), seed:int=42, samples2get:int=60_000):
    """
    Create training and validation CSVs from model predictions and ground-truth .npz files.

    This function searches recursively under `path2data` for model checkpoints
    and corresponding Train/Validation .npz files produced by the specified
    `model_name`. It loads the model, runs predictions on the final boards,
    computes an optimal per-board threshold using Youden's J statistic from
    the ROC curve between the predicted logits and ground-truth initial
    boards, removes duplicate (prediction, ground-truth) pairs, samples a
    subset of boards, splits them into training and validation sets, and
    writes two CSV files ('trainGoL.csv' and 'valGoL.csv') to `path2save`.

    Args:
        path2data (Path): Root directory where Results/Train/Validation
            subfolders are searched for model checkpoints and .npz data.
        path2save (Path): Directory where the resulting CSV files will be
            written. If not a Path object, it will be converted to Path.
        model_name (str): Subfolder name under Results/Train/Validation that
            contains model checkpoints and data (default: 'DiffGoL').
        shape (tuple): Board shape as (H, W). Used to reshape predictions
            and to build CSV column names (default: (15, 15)).
        seed (int): Random seed used when shuffling and sampling boards
            (default: 42).
        samples2get (int): Number of unique boards to sample before splitting
            into train/validation (default: 60000).

    Returns:
        dict: A dictionary with keys 'train' and 'val' whose values are the
        Paths to the created CSV files.

    Raises:
        SystemExit: If no CUDA device is available (the function requires GPU
            to run the model inference).
    """
    # Load device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print(f'❌ GPU is not detected, interrupting execution...')
        sys.exit(1)

    # Header to select
    init_cols = [f'start_{i}' for i in range(shape[0]*shape[1])]

    # Check if path2data is actually a Path:
    if not isinstance(path2data, Path):
        path2data = Path(path2data)

    preds, gtruths = [], []
    model = torch.compile(DiffGoLModel(6, 48, 3, 1, 50.0, 4.0).to(device))
    for run_dir in sorted(d for d in path2data.iterdir() if d.is_dir()):
        if f'{shape[0]}x{shape[1]}' in str(run_dir):
            model_file = run_dir / 'Results' / model_name / 'model.pt'
            train_file = run_dir / 'ConwayStates' / 'Train' / model_name / 'train0.npz'
            val_file   = run_dir / 'ConwayStates' / 'Validation' / model_name / 'val0.npz'

        if not (model_file.exists() and train_file.exists() and val_file.exists()):
            continue   # run incompleto, lo saltas

        # Load ground truth
        _, gt_train, final_train = load_npz(train_file, train_file, False) # --> (48k, 225)
        _, gt_val, final_val = load_npz(val_file, val_file, False) # --> (12k, 225)

        gts = np.vstack([gt_train, gt_val])
        gtruths.append(gts) # shape --> (60k, 225)

        # Predict with the model
        finals = np.vstack([final_train, final_val])
        
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        finals = torch.tensor(finals, dtype=torch.float32).reshape(-1, 1, shape[0], shape[1]).to(device)
        with torch.no_grad():
            init_pred, _ = model(finals)
            predictions = init_pred.cpu().numpy().reshape(-1, shape[0]*shape[1])

        preds.append(predictions) # shape (60k, 225)
        
    preds = np.vstack(preds) # shape --> (900k, 225)
    gtruths = np.vstack(gtruths) # shape --> (900k, 225)

    states = np.concatenate([preds, gtruths], axis=1) # prediction + gtruth (900k, 450)

    # Filter equal values:
    states = np.unique(states, axis=0)
    preds, gtruths = np.hsplit(states, 2)

    # Convert to tensor 
    pred = torch.tensor(preds, dtype=torch.float32)
    gtruth = torch.tensor(gtruths, dtype=torch.bool)

    # Board loop
    board_ths = []
    for b_pred, b_gtruth in zip(pred, gtruth):

        # Compute the ROC curve and the AUROC for this board
        fpr, tpr, thresholds = roc(b_pred, b_gtruth, task="binary")
        
        # To numpy
        fpr, tpr, thresholds = fpr.numpy(), tpr.numpy(), thresholds.numpy()

        # Best threshold by Youden's J statistic --> max(tpr - fpr)
        best_idx = np.argmax(tpr - fpr)
        board_ths.append(thresholds[best_idx])

    # Shuffle and select samples2get
    board_ths = np.array(board_ths)
    pred = np.array(pred)

    idx = np.arange(len(pred))
    rng = np.random.default_rng(seed) 
    rng.shuffle(idx)

    sel = idx[:samples2get]
    preds = pred[sel]
    board_ths = board_ths[sel]
    
    # Split
    x_train, x_val, y_train, y_val = train_test_split(preds, board_ths, test_size=0.2, random_state=seed)

    # Create datasets:
    if not isinstance(path2save, Path):
        path2save = Path(path2save)

    # Ensure the path exists
    path2save.mkdir(parents=True, exist_ok=True)

    df_train = pd.DataFrame(x_train, columns=init_cols)
    df_train['threshold'] = y_train
    df_train.to_csv(path2save/'trainGoL.csv', index=False)   
    
    df_val = pd.DataFrame(x_val, columns=init_cols)
    df_val['threshold'] = y_val
    df_val.to_csv(path2save/'valGoL.csv', index=False)   
    

def create_random_test(path2data:Path, path2save:Path, model:str='DiffGoL', shape:tuple=(15,15), seed:int=42, samples2get:int=15_000):
    """
    Create a CSV test dataset from model predictions and ground-truth test files.

    This function collects prediction CSVs produced by a given model and the
    corresponding test.npz files containing ground-truth boards. It computes a
    per-board optimal threshold using Youden's J statistic from the ROC curve,
    remove identical (prediction, ground-truth) pairs, samples a subset
    of boards, and saves the resulting prediction vectors with their
    thresholds to a CSV file named 'testGoL.csv' under `path2save`.

    Args:
        path2data (Path): Root path where model Results and test.npz files are
            searched recursively.
        path2save (Path): Directory where the resulting 'testGoL.csv' will be
            written.
        model (str): Model subfolder under Results to load predictions from
            (default: 'DiffGoL').
        shape (tuple): Shape of the game board as (H, W). Used to build column
            names for prediction vectors (default: (15, 15)).
        seed (int): Random seed used to shuffle and sample boards
            (default: 42).
        samples2get (int): Number of boards to sample and save to the CSV
            (default: 15000).

    Returns:
        Path to the 'testGoL.csv' file.

    Notes:
        - Expects prediction CSVs at '**/Results/{model}/predictions.csv' and
          test files at '**/test.npz' under `path2data`.
        - The output CSV contains one row per sampled board with columns
          'start_0'..'start_{H*W-1}' for predicted scores and a 'threshold'
          column with the board-specific optimal binarization threshold.
    """
    # Header to select
    init_cols = [f'start_{i}' for i in range(shape[0]*shape[1])]
    gt_cols = [f'gt_{i}' for i in range(shape[0]*shape[1])]
    # Check if path2data is actually a Path:
    if not isinstance(path2data, Path):
        path2data = Path(path2data)

    # Get the files:
    preds, gtruths = [], []
    for run_dir in sorted(d for d in path2data.iterdir() if d.is_dir()):
        if f'{shape[0]}x{shape[1]}' in str(run_dir):
            datafile = run_dir / 'Results' / model / 'predictions.csv'
            testfile = run_dir / 'ConwayStates' / 'Test' / 'test.npz'

        if not (datafile.exists() and testfile.exists()):
            continue
        
        # Load predictions
        df = pd.read_csv(datafile, sep=',')
        preds.append(df[init_cols].values) # shape --> (15k, 225)

        # Load ground truth
        _, y_true, _ = load_npz(testfile, testfile, False) # --> (15k, 225)
        gtruths.append(y_true)

    preds = np.vstack(preds) # shape --> (225k, 225)
    gtruths = np.vstack(gtruths) # shape --> (225k, 225)

    states = np.concatenate([preds, gtruths], axis=1) # prediction + gtruth (225k, 450)

    # Filter equal values:
    states = np.unique(states, axis=0)
    preds, gtruths = np.hsplit(states, 2)

    # Convert to tensor 
    pred = torch.tensor(preds, dtype=torch.float32)
    gtruth = torch.tensor(gtruths, dtype=torch.bool)

    # Board loop
    board_ths = []
    for b_pred, b_gtruth in zip(pred, gtruth):

        # Compute the ROC curve and the AUROC for this board
        fpr, tpr, thresholds = roc(b_pred, b_gtruth, task="binary")
        
        # To numpy
        fpr, tpr, thresholds = fpr.numpy(), tpr.numpy(), thresholds.numpy()

        # Best threshold by Youden's J statistic --> max(tpr - fpr)
        best_idx = np.argmax(tpr - fpr)
        board_ths.append(thresholds[best_idx])

    # Shuffle and split:
    board_ths = np.array(board_ths)
    pred = np.array(pred)
    gtruth = np.array(gtruth)

    idx = np.arange(len(pred))
    rng = np.random.default_rng(seed) 
    rng.shuffle(idx)

    sel = idx[:samples2get]
    preds = pred[sel]
    gtruth = gtruth[sel]
    board_ths = board_ths[sel]

    # Create datasets:
    if not isinstance(path2save, Path):
        path2save = Path(path2save)
    
    # Ensure the path exits
    path2save.mkdir(parents=True, exist_ok=True)

    df_test_cm = pd.DataFrame(np.concatenate([preds, gtruth], axis=1), columns=init_cols + gt_cols)
    df_test_cm.to_csv(path2save / 'confusion_matrix.csv', index=False)
    
    df_test = pd.DataFrame(preds, columns=init_cols)
    df_test['threshold'] = board_ths
    df_test.to_csv(path2save/'testGoL.csv', index=False)   

def create_test_dataset(path2data:Path, path2save:Path, model:str='DiffGoL', shape:tuple=(15,15), samples_per_seed:int=1000):
    # Headers
    init_cols = [f'start_{i}' for i in range(shape[0]*shape[1])]
    gt_cols   = [f'gt_{i}'    for i in range(shape[0]*shape[1])]

    if not isinstance(path2data, Path):
        path2data = Path(path2data)

    # Collect valid directories in deterministic order (= seed order)
    run_dirs = []
    for run_dir in sorted(d for d in path2data.iterdir() if d.is_dir()):
        datafile = run_dir / 'Results' / model / 'predictions.csv'
        testfile = run_dir / 'ConwayStates' / 'Test' / 'test.npz'
        if datafile.exists() and testfile.exists():
            run_dirs.append((datafile, testfile))

    # Take the block [i*N, (i+1)*N) from each seed
    preds_list, gt_list = [], []
    for i, (datafile, testfile) in enumerate(run_dirs):
        lo, hi = i * samples_per_seed, (i + 1) * samples_per_seed

        df = pd.read_csv(datafile, sep=',')
        seed_preds = df[init_cols].values            # (15000, 225)

        _, y_true, _ = load_npz(testfile, testfile, False)  # (15000, 225)

        preds_list.append(seed_preds[lo:hi])         # (1000, 225)
        gt_list.append(y_true[lo:hi])                # (1000, 225)

    preds   = np.vstack(preds_list)                  # (15000, 225) in order 0..14999
    gtruths = np.vstack(gt_list)                     # (15000, 225)

    # Optimal threshold per board (Youden's J)
    pred_t = torch.tensor(preds,   dtype=torch.float32)
    gt_t   = torch.tensor(gtruths, dtype=torch.bool)

    board_ths = []
    for b_pred, b_gtruth in zip(pred_t, gt_t):
        fpr, tpr, thresholds = roc(b_pred, b_gtruth, task="binary")
        fpr, tpr, thresholds = fpr.numpy(), tpr.numpy(), thresholds.numpy()
        best_idx = np.argmax(tpr - fpr)
        board_ths.append(thresholds[best_idx])
    board_ths = np.array(board_ths)

    # Save
    if not isinstance(path2save, Path):
        path2save = Path(path2save)
    path2save.mkdir(parents=True, exist_ok=True)

    df_test_cm = pd.DataFrame(np.concatenate([preds, gtruths], axis=1), columns=init_cols + gt_cols)
    df_test_cm.to_csv(path2save / 'confusion_matrix.csv', index=False)

    df_test = pd.DataFrame(preds, columns=init_cols)
    df_test['threshold'] = board_ths
    df_test.to_csv(path2save / 'testGoL.csv', index=False)

def create_train_dataset_PR(path2data:Path, path2save:Path, model_name:str='DiffGoL', shape:tuple=(15,15), seed:int=42, samples2get:int=60_000):
    """
    Create training and validation CSVs from model predictions and ground-truth .npz files.

    This function searches recursively under `path2data` for model checkpoints
    and corresponding Train/Validation .npz files produced by the specified
    `model_name`. It loads the model, runs predictions on the final boards,
    computes an optimal per-board threshold by maximizing the F1 score on the
    precision-recall curve between the predicted scores and ground-truth
    initial boards, removes duplicate (prediction, ground-truth) pairs, samples
    a subset of boards, splits them into training and validation sets, and
    writes two CSV files ('trainGoL.csv' and 'valGoL.csv') to `path2save`.

    Args:
        path2data (Path): Root directory where Results/Train/Validation
            subfolders are searched for model checkpoints and .npz data.
        path2save (Path): Directory where the resulting CSV files will be
            written. If not a Path object, it will be converted to Path.
        model_name (str): Subfolder name under Results/Train/Validation that
            contains model checkpoints and data (default: 'DiffGoL').
        shape (tuple): Board shape as (H, W). Used to reshape predictions
            and to build CSV column names (default: (15, 15)).
        seed (int): Random seed used when shuffling and sampling boards
            (default: 42).
        samples2get (int): Number of unique boards to sample before splitting
            into train/validation (default: 60000).

    Returns:
        dict: A dictionary with keys 'train' and 'val' whose values are the
        Paths to the created CSV files.

    Raises:
        SystemExit: If no CUDA device is available (the function requires GPU
            to run the model inference).
    """
    # Load device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print(f'❌ GPU is not detected, interrupting execution...')
        sys.exit(1)

    # Header to select
    init_cols = [f'start_{i}' for i in range(shape[0]*shape[1])]

    # Check if path2data is actually a Path:
    if not isinstance(path2data, Path):
        path2data = Path(path2data)

    preds, gtruths = [], []
    model = torch.compile(DiffGoLModel(6, 48, 3, 1, 50.0, 4.0).to(device))
    for run_dir in sorted(d for d in path2data.iterdir() if d.is_dir()):
        model_file = run_dir / 'Results' / model_name / 'model.pt'
        train_file = run_dir / 'ConwayStates' / 'Train' / model_name / 'train0.npz'
        val_file   = run_dir / 'ConwayStates' / 'Validation' / model_name / 'val0.npz'

        if not (model_file.exists() and train_file.exists() and val_file.exists()):
            continue   # run incompleto, lo saltas

        # Load ground truth
        _, gt_train, final_train = load_npz(train_file, train_file, False) # --> (48k, 225)
        _, gt_val, final_val = load_npz(val_file, val_file, False) # --> (12k, 225)

        gts = np.vstack([gt_train, gt_val])
        gtruths.append(gts) # shape --> (60k, 225)

        # Predict with the model
        finals = np.vstack([final_train, final_val])

        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        finals = torch.tensor(finals, dtype=torch.float32).reshape(-1, 1, shape[0], shape[1]).to(device)
        with torch.no_grad():
            init_pred, _ = model(finals)
            predictions = init_pred.cpu().numpy().reshape(-1, shape[0]*shape[1])

        preds.append(predictions) # shape (60k, 225)

    preds = np.vstack(preds) # shape --> (900k, 225)
    gtruths = np.vstack(gtruths) # shape --> (900k, 225)

    states = np.concatenate([preds, gtruths], axis=1) # prediction + gtruth (900k, 450)

    # Filter equal values:
    states = np.unique(states, axis=0)
    preds, gtruths = np.hsplit(states, 2)

    # Convert to tensor
    pred = torch.tensor(preds, dtype=torch.float32)
    gtruth = torch.tensor(gtruths, dtype=torch.bool)

    # Board loop
    board_ths = []
    for b_pred, b_gtruth in zip(pred, gtruth):

        # Compute the precision-recall curve for this board
        precision, recall, thresholds = precision_recall_curve(b_pred, b_gtruth, task="binary")

        # To numpy
        precision, recall, thresholds = precision.numpy(), recall.numpy(), thresholds.numpy()

        # Best threshold by max F1.
        # precision/recall have length n+1; thresholds has length n. The last
        # point (P=1, R=0) has no associated threshold -> exclude with [:-1].
        f1 = 2 * precision * recall / (precision + recall + 1e-12)  # epsilon avoids 0/0
        best_idx = np.argmax(f1[:-1])
        board_ths.append(thresholds[best_idx])

    # Shuffle and select samples2get
    board_ths = np.array(board_ths)
    pred = np.array(pred)

    idx = np.arange(len(pred))
    rng = np.random.default_rng(seed) 
    rng.shuffle(idx)

    sel = idx[:samples2get]
    preds = pred[sel]
    board_ths = board_ths[sel]

    # Split
    x_train, x_val, y_train, y_val = train_test_split(preds, board_ths, test_size=0.2, random_state=seed)

    # Create datasets:
    if not isinstance(path2save, Path):
        path2save = Path(path2save)

    # Ensure the path exists
    path2save.mkdir(parents=True, exist_ok=True)

    df_train = pd.DataFrame(x_train, columns=init_cols)
    df_train['threshold'] = y_train
    df_train.to_csv(path2save/'trainGoL.csv', index=False)   

    df_val = pd.DataFrame(x_val, columns=init_cols)
    df_val['threshold'] = y_val
    df_val.to_csv(path2save/'valGoL.csv', index=False)



def create_test_dataset_PR(path2data:Path, path2save:Path, model:str='DiffGoL', shape:tuple=(15,15), samples_per_seed:int=1000):
    # Headers
    init_cols = [f'start_{i}' for i in range(shape[0]*shape[1])]
    gt_cols   = [f'gt_{i}'    for i in range(shape[0]*shape[1])]

    if not isinstance(path2data, Path):
        path2data = Path(path2data)

    # Collect valid directories in deterministic order (= seed order)
    run_dirs = []
    for run_dir in sorted(d for d in path2data.iterdir() if d.is_dir()):
        datafile = run_dir / 'Results' / model / 'predictions.csv'
        testfile = run_dir / 'ConwayStates' / 'Test' / 'test.npz'
        if datafile.exists() and testfile.exists():
            run_dirs.append((datafile, testfile))

    # Take the block [i*N, (i+1)*N) from each seed
    preds_list, gt_list = [], []
    for i, (datafile, testfile) in enumerate(run_dirs):
        lo, hi = i * samples_per_seed, (i + 1) * samples_per_seed

        df = pd.read_csv(datafile, sep=',')
        seed_preds = df[init_cols].values            # (15000, 225)

        _, y_true, _ = load_npz(testfile, testfile, False)  # (15000, 225)

        preds_list.append(seed_preds[lo:hi])         # (1000, 225)
        gt_list.append(y_true[lo:hi])                # (1000, 225)

    preds   = np.vstack(preds_list)                  # (15000, 225) in order 0..14999
    gtruths = np.vstack(gt_list)                     # (15000, 225)

    # Optimal threshold per board (max F1 on the precision-recall curve)
    pred_t = torch.tensor(preds,   dtype=torch.float32)
    gt_t   = torch.tensor(gtruths, dtype=torch.bool)

    board_ths = []
    for b_pred, b_gtruth in zip(pred_t, gt_t):
        precision, recall, thresholds = precision_recall_curve(b_pred, b_gtruth, task="binary")
        precision, recall, thresholds = precision.numpy(), recall.numpy(), thresholds.numpy()

        # precision/recall have length n+1; thresholds has length n.
        # The last point (P=1, R=0) has no associated threshold -> exclude with [:-1].
        f1 = 2 * precision * recall / (precision + recall + 1e-12)  # epsilon avoids 0/0
        best_idx = np.argmax(f1[:-1])
        board_ths.append(thresholds[best_idx])
    board_ths = np.array(board_ths)

    # Save
    if not isinstance(path2save, Path):
        path2save = Path(path2save)
    path2save.mkdir(parents=True, exist_ok=True)

    df_test_cm = pd.DataFrame(np.concatenate([preds, gtruths], axis=1), columns=init_cols + gt_cols)
    df_test_cm.to_csv(path2save / 'confusion_matrix.csv', index=False)

    df_test = pd.DataFrame(preds, columns=init_cols)
    df_test['threshold'] = board_ths
    df_test.to_csv(path2save / 'testGoL.csv', index=False)