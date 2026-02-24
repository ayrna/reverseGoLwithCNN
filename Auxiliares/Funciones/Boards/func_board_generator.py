import numpy as np
import tensorflow as tf
import Auxiliares.Funciones.Boards.func_generador as builder
import Auxiliares.Funciones.Boards.func_waterfilling as seleccionador
import Auxiliares.Funciones.Boards.func_create_npz as writer
import Auxiliares.Funciones.Modelo.func_conway as conway

def board_generator(seed, delta, shape, dir_npz, name, 
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
    generador = builder.generador(num_tableros, 
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

    for tablero in bases:
        # Tableros iniciales: 3 generaciones de calentamiento
        tablero_inicial = conway.game_of_life(tablero, 3) 
        iniciales.append(tablero_inicial) 

        # Tableros finales
        tablero_final = conway.game_of_life(tablero_inicial, delta)
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
    indices = seleccionador.WaterFilling(densidades, 
                                         n_objetivo=n_objetivo, 
                                         n_bins=300, 
                                         rango=(0.05, max + 0.05)
                                         )
     
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
        writer.create_npz(delta, bases_train, iniciales_train, finales_train, dir_npz[0], name[0]) # (k, H*W), (k, H*W), (k, H*W)
        writer.create_npz(delta, bases_val, iniciales_val, finales_val, dir_npz[1], name[1])
    else:
        writer.create_npz(delta, bases, iniciales, finales, dir_npz, name) 
        

    