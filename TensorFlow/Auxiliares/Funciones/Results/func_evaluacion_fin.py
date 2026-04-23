import Auxiliares.Clases.class_GoLayer as GoLayer
import Auxiliares.Funciones.Results.func_evaluacion_init as evaluator_init
import Auxiliares.Funciones.Boards.func_load_npz as loader
import pandas as pd
import numpy as np


def evolve_init_boards(init_boards, delta=1, shape=(15, 15)):
    """
    Performs vectorized inference to evolve a batch of initial states in a single graph operation.

    Args:
        init_boards (numpy.ndarray): Array of flattened initial boards. Shape: (N_samples, H*W).
        delta (int, optional): Number of generations to simulate. Defaults to 1.
        shape (tuple[int, int], optional): Spatial dimensions (height, width) of the boards. Defaults to (15, 15).

    Returns:
        numpy.ndarray: Array of flattened evolved final boards. Shape: (N_samples, H*W).
    """
    N = init_boards.shape[0]
    H, W = shape
    
    # 1. Reshape de todo el lote a formato tensor (Batch, Height, Width, Channels)
    tensor_boards = init_boards.copy().reshape(N, H, W, 1)
    
    # 2. Instanciamos la capa UNA SOLA VEZ
    diffGoL = GoLayer.ConwayLayer(delta=delta, epsilon=50, order=4)
    
    # 3. Propagación de todo el lote simultáneamente (Aprovecha paralelización)
    final_boards = diffGoL(tensor_boards)
    
    # 4. Extraemos a NumPy y aplanamos manteniendo la dimensión del lote
    return final_boards.numpy().reshape(N, H * W)

def Sobolev(path_reales, path_predicciones, cols_names, shape, umbral, normalize, classic=False):
    """
    Evaluates final states predictions using the Sobolev norm and compares them against multiple baselines.

    Calculates the Sobolev score for both continuous (probability) and binarized predictions, 
    evaluating them against the ground truth. It also computes scores for reference baselines 
    (e.g., spatial shifts, all-zeros, all-ones) to provide context on the model's performance.

    Args:
        path_reales (dict): Mapping of prediction identifiers to their corresponding ground truth .npz file paths.
        path_predicciones (dict): Mapping of identifiers to the predicted .csv file paths.
        cols_names (list[str]): Names of the columns in the CSV files containing the flattened board probabilities.
        shape (tuple[int, int]): Spatial dimensions (height, width) of the boards.
        umbral (float): Threshold value to binarize continuous probability predictions.
        normalize (bool): If True, normalizes the Sobolev norm calculation.
        classic (bool, optional): If True, evolves the predictions through the ConwayLayer before evaluation. Defaults to False.

    Returns:
        None: Prints the evaluation metrics to standard output.
    """
    sobolevs_probs= []
    sobolevs_bins = []
    
    scores_baseline_OX = []
    scores_baseline_OY = []
    scores_targets = []
    scores_opuestos = []
    scores_allones = []
    scores_allzero = []
    scores_fuzzy_boards = []

    for path_preds in path_predicciones:

        _, _, tableros_reales = loader.load_npz(path_reales[path_preds], f'test_{path_preds}.npz')
        targets = tableros_reales.reshape(-1, 15, 15)

        df_predicciones = pd.read_csv(path_predicciones[path_preds], sep=',')

        tableros_prob = df_predicciones[cols_names].values  # (15_000, 225)
        
        if classic:
            preds_prob = evolve_init_boards(tableros_prob)
        else:
            preds_prob = tableros_prob.copy()

        preds_bin = (preds_prob > umbral).astype('float32')

        # Objetivo:
        scores_pred_bin_temp = []
        scores_pred_prob_temp = []
        # Referencias
        scores_baseline_OX_temp = []
        scores_baseline_OY_temp = []
        perfect_scores_temp = []
        scores_opuestos_temp = []
        scores_allzero_temp = []
        scores_allones_temp = []
        scores_fuzzy_boards_temp = []

        for tab_prob, tab_bin, real in zip(preds_prob, preds_bin, targets):
            
            # Objetivo:
            scores_pred_prob = evaluator_init.sobolev(tab_prob.reshape(-1, shape[0], shape[1]), real, normalize) 
            scores_pred_bin = evaluator_init.sobolev(tab_bin.reshape(-1, shape[0], shape[1]), real, normalize) 

            scores_pred_prob_temp.append(scores_pred_prob)
            scores_pred_bin_temp.append(scores_pred_bin)
            
            # Referencias:
            allzero = np.zeros(shape=(15,15))
            allones = np.ones(shape=(15, 15))
            fuzzy_board = 0.5*np.ones(shape=(15, 15))
            real_desplazados_OX = np.roll(real, shift=1, axis=-1)
            real_desplazados_OY = np.roll(real, shift=1, axis=-2)
            not_real = 1 - real

            # Casi perfecto con desplazamiento OX
            scores_baseline_OX_temp.append(evaluator_init.sobolev(real_desplazados_OX, real, normalize)) 
            # Casi perfecto con desplazamiento OY
            scores_baseline_OY_temp.append(evaluator_init.sobolev(real_desplazados_OY, real, normalize)) 
            # Si predijese todo muerto
            scores_allzero_temp.append(evaluator_init.sobolev(allzero, real, normalize)) 
            # Si predijese todo vivo (error máximo)
            scores_allones_temp.append(evaluator_init.sobolev(allones, real, normalize)) 
            # Si predijese todo vivo (error máximo)
            scores_fuzzy_boards_temp.append(evaluator_init.sobolev(fuzzy_board, real, normalize)) 
            # Si fuese perfecto
            perfect_scores_temp.append(evaluator_init.sobolev(real, real, normalize)) 
            # Si predijese justo lo contrario
            scores_opuestos_temp.append(evaluator_init.sobolev(not_real, real, normalize)) 

        sobolevs_probs.append(np.mean(scores_pred_prob_temp))
        sobolevs_bins.append(np.mean(scores_pred_bin_temp))

        scores_baseline_OX.append(np.mean(scores_baseline_OX_temp))
        scores_baseline_OY.append(np.mean(scores_baseline_OY_temp))
        scores_targets.append(np.mean(perfect_scores_temp))
        scores_opuestos.append(np.mean(scores_opuestos_temp))
        scores_allones.append(np.mean(scores_allones_temp))
        scores_allzero.append(np.mean(scores_allzero_temp))
        scores_fuzzy_boards.append(np.mean(scores_fuzzy_boards_temp))

    score_bin = np.mean(sobolevs_bins)
    score_prob = np.mean(sobolevs_probs)
    std_bin = np.std(sobolevs_bins)
    std_prob = np.std(sobolevs_probs)
    print('======'*10)
    if np.mean(scores_targets) != 0:
        print('Error, Sobolev de targets sobre targets no es cero')
    else:
        print(f"Sobolev medio modelo(prob): {score_prob:.4f} ± {std_prob:.4f} ")
        print(f"Sobolev medio modelo(bin): {score_bin:.4f} ± {std_bin:.4f}")
        print(f"Sobolev medio baseline OX: {np.mean(scores_baseline_OX):.4f}")
        print(f"Sobolev medio baseline OY: {np.mean(scores_baseline_OY):.4f}")
        print(f"Sobolev medio all zeros: {np.mean(scores_allzero):.4f}")
        print(f"Sobolev medio all ones: {np.mean(scores_allones):.4f}")
        print(f"Sobolev medio si inversión: {np.mean(scores_opuestos):.4f}")
        print(f"Sobolev medio si fuzzy boards: {np.mean(scores_fuzzy_boards):.4f}")

    performance_gap_OX =  np.abs(np.array(scores_baseline_OX) - np.array(sobolevs_bins))
    print(f"Diferencia Media en OX (bin): {np.mean(performance_gap_OX):.4f}")
    performance_gap_OY =  np.abs(np.array(scores_baseline_OY) - np.array(sobolevs_bins))
    print(f"Diferencia Media en OY (bin): {np.mean(performance_gap_OY):.4f}")
    performance_gap_OX =  np.abs(np.array(scores_baseline_OX) - np.array(sobolevs_probs))
    print(f"Diferencia Media en OX (prob): {np.mean(performance_gap_OX):.4f}")
    performance_gap_OY =  np.abs(np.array(scores_baseline_OY) - np.array(sobolevs_probs))
    print(f"Diferencia Media en OY (prob): {np.mean(performance_gap_OY):.4f}")


def fuzziness_index(path_predicciones, cols_names, classic=False):
    """
    Computes and prints the fuzziness index of the final states predictions.

    The fuzziness index quantifies the uncertainty or lack of sharpness in the 
    continuous probability predictions across the dataset.

    Args:
        path_predicciones (dict): Mapping of identifiers to the predicted .csv file paths.
        cols_names (list[str]): Names of the columns in the CSV files containing the flattened board probabilities.
        classic (bool, optional): If True, evolves the predictions through the ConwayLayer before evaluation. Defaults to False.

    Returns:
        None: Prints the calculated mean and standard deviation of the fuzziness index.
    """
    fuzzines_values = []
    fuzzines_std = []

    for path_preds in path_predicciones:
        
        temp_fuzz = []
        
        # Carga de datos
        df_predicciones = pd.read_csv(path_predicciones[path_preds], sep=',')
        tableros_prob = df_predicciones[cols_names].values

        if classic:
            preds_prob = evolve_init_boards(tableros_prob)
        else:
            preds_prob = tableros_prob.copy()
        
        # --- BUCLE DE MUESTRAS (Tablero a Tablero) ---
        for tab_prob in preds_prob:
            borrosidad = evaluator_init.indice_borrosidad(tab_prob.reshape(-1, 15, 15))
            temp_fuzz.append(borrosidad)

        model_mean = np.mean(temp_fuzz)
        model_std = np.std(temp_fuzz)

        fuzzines_values.append(model_mean)
        fuzzines_std.append(model_std)

    print('====='*10)
    print(f"Fuzziness Index: {np.mean(fuzzines_values):.4f} ± {np.std(fuzzines_std):.4f}")

def MSE(path_predicciones, path_reales, cols_names, umbral, classic=True):
    """
    Calculates and prints the Mean Squared Error (MSE) of the final states predictions.

    Computes the MSE for both continuous probability predictions and binarized predictions 
    against the flattened ground truth arrays. It also evaluates an all-zeros baseline.

    Args:
        path_predicciones (dict): Mapping of identifiers to the predicted .csv file paths.
        path_reales (dict): Mapping of prediction identifiers to their corresponding ground truth .npz file paths.
        cols_names (list[str]): Names of the columns in the CSV files containing the flattened board probabilities.
        umbral (float): Threshold value to binarize continuous probability predictions.
        classic (bool, optional): If True, evolves the predictions through the ConwayLayer before evaluation. Defaults to True.

    Returns:
        None: Prints the MSE metrics for probabilities, binarized outputs, and the all-zeros baseline.
    """
    model_mse_prob = []
    model_mse_bin = []
    allzeros = []
    for path_preds in path_predicciones:
        
        current_mse_prob = []
        current_mse_bin = []
        current_all_zeros = []
        all_zeros = np.zeros((15,15))

        # Carga de datos
        try:
            _, _, tableros_reales = loader.load_npz(path_reales[path_preds], f'test_{path_preds}.npz')
            df_predicciones = pd.read_csv(path_predicciones[path_preds], sep=',')
        except Exception as e:
            print(f"Error cargando {path_preds}: {e}")
            continue

        tableros_prob = df_predicciones[cols_names].values
        if classic:
            preds_prob = evolve_init_boards(tableros_prob)
        else:
            preds_prob = tableros_prob.copy()

        preds_bin = preds_prob > umbral

        # Iteración por muestras
        for tab_bin, tab_prob, real in zip(preds_bin, preds_prob, tableros_reales):
            
            real_flat = real.reshape(-1, 15, 15).ravel()
            prob_flat = tab_prob.reshape(-1, 15, 15).ravel()
            bin_flat = tab_bin.astype('float32').ravel() 
            all_zeros = all_zeros.ravel()

            # Probabilidad
            mse_p = np.mean((real_flat - prob_flat)**2)
            current_mse_prob.append(mse_p)

            # Binario
            mse_b = np.mean((real_flat - bin_flat)**2)
            current_mse_bin.append(mse_b)

            # Allzeros
            mse_b = np.mean((real_flat - all_zeros)**2)
            current_all_zeros.append(mse_b)
            
        model_mse_prob.append(np.nanmean(current_mse_prob))
        model_mse_bin.append(np.nanmean(current_mse_bin))
        allzeros.append(np.nanmean(current_all_zeros))

    print('====='*10)
    print(f"MSE Medio  (prob): {np.mean(model_mse_prob):.4f} ± {np.std(model_mse_prob):.4f}")
    print(f"MSE Medio (bin): {np.mean(model_mse_bin):.4f} ± {np.std(model_mse_bin):.4f}")
    print(f"MSE Medio allzeros: {np.mean(allzeros):.4f} ± {np.std(allzeros):.4f}")


def compute_Wasserstein(path_predicciones, path_reales, cols_names, classic=True):
    """
    Computes and prints the Wasserstein (Earth Mover's Distance) metric for the final states predictions.

    Utilizes a precomputed cost matrix to evaluate the spatial displacement cost between the 
    predicted probability distributions and the ground truth boards.

    Args:
        path_predicciones (dict): Mapping of identifiers to the predicted .csv file paths.
        path_reales (dict): Mapping of prediction identifiers to their corresponding ground truth .npz file paths.
        cols_names (list[str]): Names of the columns in the CSV files containing the flattened board probabilities.
        classic (bool, optional): If True, evolves the predictions through the ConwayLayer before evaluation. Defaults to True.

    Returns:
        None: Prints the mean and standard deviation of the computed Wasserstein distance.
    """
    cost_matrix = evaluator_init.build_cost_matrix(15)

    EMDs_means = []
    for path_preds in path_predicciones:
        temp_emd = []
        
        # Carga de datos
        _, tableros_reales, _ = loader.load_npz(path_reales[path_preds], f'test_{path_preds}.npz')
        targets = tableros_reales.reshape(-1, 15, 15)
        
        df_predicciones = pd.read_csv(path_predicciones[path_preds], sep=',')
        tableros_prob = df_predicciones[cols_names].values
        if classic:
            preds_prob = evolve_init_boards(tableros_prob)
        else:
            preds_prob = tableros_prob.copy()
        

        # Bucle de evaluación tablero a tablero
        for tab_prob, tab_real in zip(preds_prob, targets):
            # 2. Pasamos la matriz precalculada
            emd = evaluator_init.wasserstein(tab_real, tab_prob.reshape(-1, 15, 15), cost_matrix)
            temp_emd.append(emd)

        # Guardamos la media de esta semilla en concreto
        model_mean = np.mean(temp_emd)
        EMDs_means.append(model_mean)

    print('====='*10)
    print(f"Wasserstein (EMD): {np.mean(EMDs_means):.4f} ± {np.std(EMDs_means):.4f}")