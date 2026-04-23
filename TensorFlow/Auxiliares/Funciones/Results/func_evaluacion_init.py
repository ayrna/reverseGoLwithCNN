import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import Auxiliares.Funciones.Graficas.func_graficador as grapher
import Auxiliares.Funciones.Boards.func_load_npz as loader
import Auxiliares.Funciones.Modelo.func_conway as conway
import Auxiliares.Clases.class_GoLayer as GoLayer
import ot
from scipy.spatial.distance import cdist

########################## PUNTO 1: MÉTRICAS TRAIN ##########################
def loss_train_val(path_historiales):
    """
    Plots the mean and standard deviation of training and validation loss across multiple runs.

    Aggregates the loss data from multiple CSV history files to compute epoch-wise statistics,
    and visualizes the loss curves with standard deviation bands.

    Args:
        path_historiales (list[str]): List of file paths to the CSV training histories.

    Returns:
        None: Displays the loss plot and prints the final mean and standard deviation.
    """
    # Contenedor para dataframes
    dfs_train = []
    dfs_val = []
    for i, path_history in enumerate(path_historiales):
        df = pd.read_csv(path_history)

        # Índice como 'epoch'
        dfs_train.append(df['loss'].rename(f'run_{i}'))
        dfs_val.append(df['val_loss'].rename(f'run_{i}'))

    # Concatenamos horizontalmente (axis=1) (Cada fila es una epoch)
    df_train = pd.concat(dfs_train, axis=1)
    df_val = pd.concat(dfs_val, axis=1)

    # Cálculo de media y desviación estándar:
    mean_train = df_train.mean(axis=1)
    std_train = df_train.std(axis=1)
    mean_val = df_val.mean(axis=1)
    std_val = df_val.std(axis=1)

    # Epochs
    epochs = df_train.shape[0]
    x = np.arange(0, epochs)
    print(f'Loss en train: {mean_train.iloc[-1]:.4f} ± {std_train.iloc[-1]:.4f}')
    print(f'Loss en val: {mean_val.iloc[-1]:.4f} ± {std_val.iloc[-1]:.4f}\n')

    plt.figure(figsize=(10, 6))
    # Entrenamiento
    plt.plot(x, mean_train, color='blue', label='Train Mean')
    plt.fill_between(x, mean_train - std_train, mean_train + std_train, 
                    color='blue', alpha=0.2, label='Train Std Dev')

    # Validación
    plt.plot(x, mean_val, color='red', label='Val Mean')
    plt.fill_between(x, mean_val - std_val, mean_val + std_val, 
                    color='red', alpha=0.2, label='Val Std Dev')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss (Mean ± Std)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# -----------------------------------------------------------------------------------------
  
def metrics_train_val(path_historiales, type):
    """
    Plots training and validation metrics (mean ± std) across epochs for multiple runs.

    Dynamically adjusts the plotted metrics and subplot grid based on whether the 
    evaluation focuses on the 'init' (initial states) or 'final' (evolved states) stage.

    Args:
        path_historiales (list[str]): List of file paths to the CSV training histories.
        type (str): Type of metrics to plot. Expected values: 'init' or 'final'.

    Returns:
        None: Displays the metrics subplot grid.
    """
    if (type == 'init'):
        train = ['accuracy', 'recall', 'precision', 'specificity', 'F1score', 'absolute']
        val = ['val_accuracy', 'val_recall', 'val_precision', 'val_specificity', 'val_F1score', 'val_absolute']
        headers = ['Accuracy', 'Recall', 'Precision', 'Specificity', 'F1score', 'absolute']
        fig, ax = plt.subplots(2, 3, figsize=(20,10))
    elif (type == 'final'):
        train = ['acc_fin', 'recall_fin', 'precision_fin', 'specificity_fin' ]
        val = ['val_acc_fin', 'val_recall_fin', 'val_precision_fin', 'val_specificity_fin']
        headers = ['Accuracy', 'Recall', 'Precision', 'Specificity' ]
        fig, ax = plt.subplots(1,4, figsize=(20,5))

    
    ax = ax.flatten()
    for j in range(len(train)):
        # Contenedor para dataframes
        dfs_train = []
        dfs_val = []
        for i, path_history in enumerate(path_historiales):
            df = pd.read_csv(path_history)

            # Índice como 'epoch'
            dfs_train.append(df[train[j]].rename(f'run_{i}'))
            dfs_val.append(df[val[j]].rename(f'run_{i}'))
            
        # Concatenamos horizontalmente (axis=1) (Cada fila es una epoch)
        df_train = pd.concat(dfs_train, axis=1)
        df_val = pd.concat(dfs_val, axis=1)

        # Cálculo de media y desviación estándar:
        mean_train = df_train.mean(axis=1)
        std_train = df_train.std(axis=1)
        mean_val = df_val.mean(axis=1)
        std_val = df_val.std(axis=1)

        # Epochs
        epochs = df_train.shape[0]
        x = np.arange(0, epochs)

        print(f'{headers[j]} en train ({type}): {mean_train.iloc[-1]:.4f} ± {std_train.iloc[-1]:.4f}')
        print(f'{headers[j]} en val ({type}): {mean_val.iloc[-1]:.4f} ± {std_val.iloc[-1]:.4f}\n')

        # Entrenamiento
        ax[j].plot(x, mean_train, color='blue', label='Train Mean')
        ax[j].fill_between(x, mean_train - std_train, mean_train + std_train, 
                        color='blue', alpha=0.2, label='Train Std Dev')

        # Validación
        ax[j].plot(x, mean_val, color='red', label='Val Mean')
        ax[j].fill_between(x, mean_val - std_val, mean_val + std_val, 
                        color='red', alpha=0.2, label='Val Std Dev')
        
        #Grafiteo
        ax[j].set_xlabel('Epochs')
        ax[j].set_ylabel(train[j])
        ax[j].set_title(f'{train[j]} (Mean ± Std)')
        ax[j].legend(loc='best')
        ax[j].grid(True, alpha=0.3)

########################## PUNTO 2: MÉTRICAS TEST ##########################
def metrics_test(path_test_results, headers, names):
    """
    Extracts and prints the aggregated test metrics (mean ± std) from multiple evaluation runs.

    Args:
        path_test_results (list[str]): List of file paths to the CSV test results.
        headers (list[str]): Column headers in the CSV files corresponding to the metrics.
        names (list[str]): Descriptive names of the metrics for console output.

    Returns:
        None: Prints the calculated statistics to the console.
    """

    for label, name in zip(headers, names):
        dfs_train = []
        for i, path_test_result in enumerate(path_test_results):
            if os.path.isfile(path_test_result):
                df = pd.read_csv(path_test_result)

                # Índice como 'epoch'
                dfs_train.append(df[label].rename(f'run_{i}'))
            

        # Concatenamos horizontalmente (axis=1) (Cada fila es una epoch)
        df_train = pd.concat(dfs_train, axis=1)
        
        # Cálculo de media y desviación estándar:
        media = df_train.mean(axis=1).iloc[-1]
        std_dev = df_train.std(axis=1).iloc[-1]
        
        if 'fin' in name or 'BCE' in name:
            print(f'{name} (finales): {media:.4f} ± {std_dev:.4f}')
        else:
            print(f'{name} (iniciales): {media:.4f} ± {std_dev:.4f}')

########################## PUNTO 3: PLOT TABLEROS ##########################
def multiplots(tabs2plot, titles, rows, cols, figsize, random, fig_title=None):
    """
    Renders a synchronized grid of subplots showing different categories of boards.

    Handles synchronization across different subfigures (e.g., matching the same board index 
    across 'Real', 'Predicted Probs', and 'Predicted Bin') either sequentially or randomly.

    Args:
        tabs2plot (list[numpy.ndarray]): List of board arrays to plot in each subfigure.
        titles (list[str]): Titles for each subfigure.
        rows (int): Number of rows in the subplot grids.
        cols (int): Number of columns in the subplot grids.
        figsize (tuple[int, int]): Dimensions of the entire figure window.
        random (bool): If True, selects boards randomly using a synchronized seed.
        fig_title (str, optional): Main title for the entire figure. Defaults to None.

    Returns:
        None: Displays the composite figure.
    """

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    subfigs = fig.subfigures(2, 3, wspace=0.07, hspace=0.07)
    
    if fig_title != None:
        fig.suptitle(f'{fig_title}', fontsize=14, color='darkblue')

    
    n_samples = len(tabs2plot[0]) 
    n_plots_per_subfig = rows * cols
    
    shared_indices = None

    if random:
        rng = np.random.default_rng()
        limit = min(n_samples, n_plots_per_subfig)
        
        shared_indices = rng.choice(n_samples, size=limit, replace=False)
    # --------------------------------

    for i, subfig in enumerate(subfigs.flat):
        subfig.suptitle(titles[i], fontsize=14, color='darkblue')
        subfig.set_facecolor('whitesmoke')
        current_axs = subfig.subplots(nrows=rows, ncols=cols)
        
        grapher.multiGray(
            tableros=tabs2plot[i],        
            ncols=rows, 
            nrows=cols, 
            shape=(15, 15),           
            axs=current_axs,
            random=False,           
            forced_indices=shared_indices 
        )

    plt.show()

# ----------------------------------------------------------------------------------------------------

def evolution(index2plot, init_bin, init_prob, tab_iniciales, tab_finales, umbral, figsize, type):
    """
    Visualizes the transition of a specific board, comparing real and predicted states.

    Depending on the 'type', it plots either the initial state predictions (binary vs. 
    probabilities vs. real) or the final evolved states by running the predictions through 
    both a continuous differential layer and standard Conway rules.

    Args:
        index2plot (int): Index of the specific board to visualize.
        init_bin (numpy.ndarray): Array of predicted initial binary boards.
        init_prob (numpy.ndarray): Array of predicted initial probability boards.
        tab_iniciales (numpy.ndarray): Array of ground truth initial boards.
        tab_finales (numpy.ndarray): Array of ground truth final boards.
        umbral (float): Threshold used for binarizing continuous predictions.
        figsize (tuple[int, int]): Dimensions of the figure window.
        type (str): Visualization focus. Expected values: 'init' or 'final'.

    Returns:
        None: Displays the comparative plots.
    """
    # Jugamos con Conway Real:
    inicial_bin = init_bin[index2plot].reshape(15,15)
    final_x_Conway = conway.game_of_life(inicial_bin)

    # Jugamos con Conway Continuo:
    capa =  GoLayer.ConwayLayer(delta=1, epsilon=50, order=4)

    inicial_prob = init_prob[index2plot].reshape(1,15,15,1)
    inicial_prueba_bin = init_bin[index2plot].reshape(1,15,15,1)
    final_continuo_prob= capa(inicial_prob)
    final_continuo_bin = capa(inicial_prueba_bin)

    # Tableros Reales:
    inicial_real = tab_iniciales[index2plot].reshape(15,15)
    final_real = tab_finales[index2plot].reshape(15,15)

    # Gráficos:
    if type == 'init':
        iniciales2plot = [inicial_real, inicial_prob[0,:,:,0], inicial_bin]
        titles_iniciales = ['Inicial real', 'Inicial predicho probs', f'Inicial predicho bin (>{umbral})']
        _ ,ax = plt.subplots(1,3, figsize=figsize)
        ax = ax.flatten()

        for i in range(3):
            ax[i].imshow(iniciales2plot[i], cmap='gray')
            ax[i].set_title(titles_iniciales[i])
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        plt.show()

    elif type=='final':
        finales2plot = [final_real, final_continuo_prob[0,:,:,0], final_continuo_prob[0,:,:,0] > umbral,  final_continuo_bin[0,:,:,0], final_x_Conway]
        titles_finales = ['Final real\n', 'Final predicho \n Conway Continuo probs (Modelo)','Final predicho \n Conway Continuo bin (Modelo)', f'Final predicho \n Conway continuo bin (>{umbral})', f'Final predicho \n Conway discreto bin (>{umbral})' ]
        fig,ax = plt.subplots(1,5, figsize=figsize)
        ax = ax.flatten()

        for i in range(5):
            ax[i].imshow(finales2plot[i], cmap='gray')
            ax[i].set_title(titles_finales[i])
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        plt.show()

########################## PUNTO 4: DENSIDADES ##########################
def density(path_predicciones, tableros, cols_name):
    """
    Calculates and compares the global density of alive cells between predictions and targets.

    Args:
        path_predicciones (dict): Mapping of identifiers to predicted .csv file paths.
        tableros (numpy.ndarray): Ground truth boards to compare against.
        cols_name (list[str]): Column names containing the flattened board probabilities.

    Returns:
        None: Prints the average densities, standard deviations, and mean absolute errors.
    """
    dens_pred = []
    stds_pred = []
    for path_preds in path_predicciones:
        df_predicciones = pd.read_csv(path_predicciones[path_preds], sep=',')

        pred = df_predicciones[cols_name]  # (15_000, shape[0], shape[1])

        # EN PROBABILIDADES
        y_pred = pred
    
        densidades_pred = np.mean(y_pred, axis=1)
        standard_pred = densidades_pred.std(ddof=1)

        dens_pred.append(densidades_pred)
        stds_pred.append(standard_pred)

    den_pred = np.mean(dens_pred)
    std_pred = np.mean(stds_pred)

    y_true = tableros
    densidades_true = np.mean(y_true, axis=1)
    den_real = densidades_true.mean()
    std_real = densidades_true.std(ddof=1)

    delta_den = np.abs(den_pred - den_real)
    delta_std = np.abs(std_pred - std_real)
    print(f'Densidad media predicha: {den_pred:.2%}')
    print(f'Densidad media real: {den_real:.2%}')
    print(f'Desviación estándar pred: {std_pred:.2%}')
    print(f'Desviación estándar real: {std_real:.2%}')
    print(f'Error en la densidad: {delta_den:.2%}')
    print(f'Error en la std: {delta_std:.2%}')

    error_medio_tab = np.mean(np.abs(densidades_pred - densidades_true))
    print(f'Error medio por tablero: {error_medio_tab:.2%}')

# -----------------------------------------------------------------------
def histograma(path_predicciones, tableros, cols_name, low_th, high_th):
    """
    Plots a cumulative logarithmic histogram of predicted pixel values vs. real values.

    Calculates the percentage of 'fuzzy' or doubtful cell predictions falling within 
    a specified probability threshold range.

    Args:
        path_predicciones (dict): Mapping of identifiers to predicted .csv file paths.
        tableros (numpy.ndarray): Ground truth boards.
        cols_name (list[str]): Column names containing the predicted probabilities.
        low_th (float): Lower bound of the 'doubtful' probability range.
        high_th (float): Upper bound of the 'doubtful' probability range.

    Returns:
        None: Displays the logarithmic histogram and prints the doubtful cell statistics.
    """
    df_list = []
    dudosos_con_std = []
    for seed in path_predicciones:
        df = pd.read_csv(path_predicciones[seed], sep=',')
        Y_pred = df[cols_name].values.flatten()
        mask_Dudosos = (Y_pred > low_th) & (Y_pred < high_th)
        count_Dudosos = np.sum(mask_Dudosos)
        dudosos_con_std.append(count_Dudosos/Y_pred.size)
        df_list.append(df)
    
    media_dudosos_con_std = np.mean(dudosos_con_std)
    std_dudosos_con_std = np.std(dudosos_con_std)

    df_total = pd.concat(df_list, axis=0)

    y_pred = df_total[cols_name].values.flatten()

    # 2. Obtenemos los valores reales
    y_true = tableros.flatten()
    # Máscara:
    mask_dudosos = (y_pred > low_th) & (y_pred < high_th)
    total = y_pred.size 
    count_dudosos = np.sum(mask_dudosos)

    porcentaje_dudosos = count_dudosos / total
    print('====='*10)
    print(f'Células totales: {total}')
    print(f'%HUC: {media_dudosos_con_std:.2%} ± {std_dudosos_con_std:.2%}')
    print(f'Porcentaje de células dudosas: {porcentaje_dudosos:.2%} ({porcentaje_dudosos * total:.0f} células)')
    # 3. Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(y_true, bins=30, color='blue', alpha=0.5, edgecolor='black',
            label='Reales', density=True)

    plt.hist(y_pred, bins=30, color='darkorange', alpha=0.5, edgecolor='black',
            label='Predicciones (Acumulado)', density=True)
    print(f'Valor máximo en las predicciones: {np.max(y_pred)}')

    plt.yscale('log')
    plt.title('Histograma acumulado')
    plt.ylabel('Densidad de Frecuencia (Log scale)')
    plt.xlabel('Valor del píxel')
    plt.legend()
    plt.show()

########################## PUNTO 5: METRICAS ##########################
def indice_borrosidad(y_pred):
    """
    Calculates the fuzziness index of a continuous probability prediction.

    The index is based on the scaled variance of the pixel probabilities, reaching 
    its maximum when probabilities are exactly 0.5 (maximum uncertainty).

    Args:
        y_pred (numpy.ndarray): Array of continuous probability values.

    Returns:
        float: The computed fuzziness index.
    """
    # Asegurar que los valores estén estrictamente en [0, 1] por seguridad numérica
    y_pred_clipped = np.clip(np.asarray(y_pred), 0.0, 1.0)
    
    # Cálculo vectorizado: media de la varianza escalada
    return float(np.mean(4.0 * y_pred_clipped * (1.0 - y_pred_clipped)))

def fuzzines_index(path_predicciones, cols_names):
    """
    Computes and prints the mean fuzziness index across multiple datasets.

    Args:
        path_predicciones (dict): Mapping of identifiers to predicted .csv file paths.
        cols_names (list[str]): Column names containing the predicted probabilities.

    Returns:
        None: Prints the mean and standard deviation of the fuzziness index.
    """
    fuzzines_values = []
    fuzzines_std = []

    for path_preds in path_predicciones:
        
        temp_fuzz = []
        
        # Carga de datos
    
        df_predicciones = pd.read_csv(path_predicciones[path_preds], sep=',')
        init_prob = df_predicciones[cols_names].values
        
        preds_prob = init_prob.copy().reshape(-1, 15, 15)

        # --- BUCLE DE MUESTRAS (Tablero a Tablero) ---
        for tab_prob in preds_prob:
            borrosidad = indice_borrosidad(tab_prob)
            temp_fuzz.append(borrosidad)

        model_mean = np.mean(temp_fuzz)
        model_std = np.std(temp_fuzz)

        fuzzines_values.append(model_mean)
        fuzzines_std.append(model_std)

    print('====='*10)
    print(f"Fuzziness Index: {np.mean(fuzzines_values):.4f} ± {np.std(fuzzines_std):.4f}")

def sobolev(pred, target, normalized=False):
    """
    Calculates the Sobolev norm of the error between a prediction and a target using 2D FFT.

    This metric penalizes errors in high-frequency domains, capturing structural and 
    spatial discrepancies better than standard pixel-wise differences.

    Args:
        pred (numpy.ndarray): Predicted spatial tensor.
        target (numpy.ndarray): Ground truth spatial tensor.
        normalized (bool, optional): If True, scales the error by the image size. Defaults to False.

    Returns:
        numpy.ndarray or float: The computed Sobolev score(s).
    """
    # Aseguro que son arrays de float
    pred = np.array(pred, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    
    # Calcular la diferencia 
    diff = target - pred
    
    # Imágenes Usamos las últimas 2 dimensiones (Mira la definición)
    H, W = diff.shape[-2:] 
    
    # FFT 2D
    fft_diff = np.fft.fft2(diff, axes=(-2, -1)) # FFT2D con H y W
    
    # Frecuencias 2D
    ky = np.fft.fftfreq(H)
    kx = np.fft.fftfreq(W)
    
    # Crear Grid: indexi='ij' para que coincida con H,W
    k_grid_y, k_grid_x = np.meshgrid(ky, kx, indexing='ij')
    
    # |k|^2 = kx^2 + ky^2
    k_squared = k_grid_x**2 + k_grid_y**2

    # |F(A-B)|^2
    energy_spectrum = np.abs(fft_diff)**2
    
    # Calcular los Pesos Sobolev 
    # W = 1 / (1 + (2*pi*|k|)^2)
    weights = 1.0 / (1.0 + (2 * np.pi)**2 * k_squared)
    
    # 5. Suma Ponderada
    weighted_error = np.sum(energy_spectrum * weights, axis=(-2,-1))
    
    # Normalizamos por el tamaño para que el error no explote con imágenes grandes
    if normalized:
        # Factor de normalización de energía (Parseval 2D)
        norm_factor = H * W
        scores = (weighted_error /norm_factor)**(1/2)
    else:
        scores = (weighted_error)**(1/2)

    return scores

def Sobolev(path_reales, path_predicciones, cols_names, shape, umbral, normalize=False):
    """
    Evaluates initial predictions against ground truth using the Sobolev norm.

    Calculates scores for continuous and binarized predictions, comparing them to 
    spatial baselines (OX/OY shifts), structural extremes (all zeros/ones), and inversions.

    Args:
        path_reales (dict): Mapping of prediction identifiers to ground truth .npz file paths.
        path_predicciones (dict): Mapping of identifiers to predicted .csv file paths.
        cols_names (list[str]): Column names containing the predicted probabilities.
        shape (tuple[int, int]): Spatial dimensions (height, width) of the boards.
        umbral (float): Threshold to binarize predictions.
        normalize (bool, optional): Normalizes the Sobolev calculation. Defaults to False.

    Returns:
        None: Prints the Sobolev metrics and performance gaps against baselines.
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

        _, tableros_reales, _ = loader.load_npz(path_reales[path_preds], f'test_{path_preds}.npz')
        targets = tableros_reales.reshape(-1, 15, 15)

        df_predicciones = pd.read_csv(path_predicciones[path_preds], sep=',')

        init_prob = df_predicciones[cols_names].values  # (15_000, shape[0], shape[1])
        
        preds_prob = init_prob.copy().reshape(-1, shape[0], shape[1])
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
            scores_pred_prob = sobolev(tab_prob, real, normalize) 
            scores_pred_bin = sobolev(tab_bin, real, normalize) 

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
            scores_baseline_OX_temp.append(sobolev(real_desplazados_OX, real, normalize)) 
            # Casi perfecto con desplazamiento OY
            scores_baseline_OY_temp.append(sobolev(real_desplazados_OY, real, normalize)) 
            # Si predijese todo muerto
            scores_allzero_temp.append(sobolev(allzero, real, normalize)) 
            # Si predijese todo vivo (error máximo)
            scores_allones_temp.append(sobolev(allones, real, normalize)) 
            # Si predijese todo vivo (error máximo)
            scores_fuzzy_boards_temp.append(sobolev(fuzzy_board, real, normalize)) 
            # Si fuese perfecto
            perfect_scores_temp.append(sobolev(real, real, normalize)) 
            # Si predijese justo lo contrario
            scores_opuestos_temp.append(sobolev(not_real, real, normalize)) 

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
# ---------------------------------------------------------------------------------------------------
def MSE(path_predicciones, path_reales, cols_names, umbral):
    """
    Computes the Mean Squared Error (MSE) for model predictions on initial states.

    Evaluates the MSE for both continuous probabilities and binarized outcomes, 
    comparing them against the ground truth and an all-zeros baseline.

    Args:
        path_predicciones (dict): Mapping of identifiers to predicted .csv file paths.
        path_reales (dict): Mapping of prediction identifiers to ground truth .npz file paths.
        cols_names (list[str]): Column names containing the predicted probabilities.
        umbral (float): Threshold to binarize predictions.

    Returns:
        None: Prints the aggregated MSE metrics.
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
            _, tableros_reales, _ = loader.load_npz(path_reales[path_preds], f'test_{path_preds}.npz')
            df_predicciones = pd.read_csv(path_predicciones[path_preds], sep=',')
        except Exception as e:
            print(f"Error cargando {path_preds}: {e}")
            continue

        init_prob = df_predicciones[cols_names].values
        preds_prob = init_prob.copy().reshape(-1, 15, 15)
        preds_bin = preds_prob > umbral

        # Iteración por muestras
        for tab_bin, tab_prob, real in zip(preds_bin, preds_prob, tableros_reales):
            
            real_flat = real.ravel()
            prob_flat = tab_prob.ravel()
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
# ----------------------------------------------------------------------------------------------------
def build_cost_matrix(grid_size=15):
    """
    Precomputes the spatial cost matrix for the Wasserstein distance calculation.

    Generates a matrix of squared Euclidean distances between all pairs of coordinates 
    in a 2D grid.

    Args:
        grid_size (int, optional): The side length of the square grid. Defaults to 15.

    Returns:
        numpy.ndarray: The precomputed cost matrix.
    """
    # Mapeo de coordenadas 2D - 1D
    coords = np.array([[i, j] for i in range(grid_size) for j in range(grid_size)])
    # Precio de la distancia: a más distancia más error ---> dist_euclidean **2
    cost_matrix = cdist(coords, coords, metric='sqeuclidean')
    return cost_matrix

def wasserstein(y_true_2d, y_pred_2d, cost_matrix, max_penalty=20.0):
    """
    Calculates the 2D Wasserstein (Earth Mover's) distance between prediction and ground truth.

    Treats the 2D boards as probability mass functions and computes the optimal transport 
    cost required to transform the predicted distribution into the true distribution.

    Args:
        y_true_2d (numpy.ndarray): Ground truth 2D board.
        y_pred_2d (numpy.ndarray): Predicted 2D probability board.
        cost_matrix (numpy.ndarray): Precomputed spatial cost matrix.
        max_penalty (float, optional): Penalty applied if one board has mass and the other is empty. Defaults to 20.0.

    Returns:
        float: The square root of the Earth Mover's Distance.
    """
    
    # Casteo recomendado
    mass_true = np.asarray(y_true_2d).flatten().astype(np.float64)
    mass_pred = np.asarray(y_pred_2d).flatten().astype(np.float64)
    
    # Calculp de masas
    sum_true = mass_true.sum()
    sum_pred = mass_pred.sum()
    
    # Manejo de Casos Límite
    if sum_true == 0 and sum_pred == 0:
        return 0.0 
    elif sum_true == 0 or sum_pred == 0:
        return max_penalty 
        
    # Normalización a Funciones de Densidad de Probabilidad
    mass_true_norm = mass_true / sum_true
    mass_pred_norm = mass_pred / sum_pred
    
    # Cálculo del Transporte Óptimo
    emd_distance = ot.emd2(mass_pred_norm, mass_true_norm, cost_matrix)
    
    return np.sqrt(emd_distance)

def compute_Wasserstein(path_predicciones, path_reales, cols_names):
    """
    Computes and prints the average Wasserstein (EMD) metric across datasets.

    Args:
        path_predicciones (dict): Mapping of identifiers to predicted .csv file paths.
        path_reales (dict): Mapping of prediction identifiers to ground truth .npz file paths.
        cols_names (list[str]): Column names containing the predicted probabilities.

    Returns:
        None: Prints the mean and standard deviation of the Wasserstein distances.
    """
    cost_matrix = build_cost_matrix(15)

    EMDs_means = []
    for path_preds in path_predicciones:
        temp_emd = []
        
        # Carga de datos
        _, tableros_reales, _ = loader.load_npz(path_reales[path_preds], f'test_{path_preds}.npz')
        targets = tableros_reales.reshape(-1, 15, 15)
        
        df_predicciones = pd.read_csv(path_predicciones[path_preds], sep=',')
        init_prob = df_predicciones[cols_names].values
        preds_prob = init_prob.copy().reshape(-1, 15, 15)

        # Bucle de evaluación tablero a tablero
        for tab_prob, tab_real in zip(preds_prob, targets):
            # 2. Pasamos la matriz precalculada
            emd = wasserstein(tab_real, tab_prob, cost_matrix)
            temp_emd.append(emd)

        # Guardamos la media de esta semilla en concreto
        model_mean = np.mean(temp_emd)
        EMDs_means.append(model_mean)

    print('====='*10)
    print(f"Wasserstein (EMD): {np.mean(EMDs_means):.4f} ± {np.std(EMDs_means):.4f}")