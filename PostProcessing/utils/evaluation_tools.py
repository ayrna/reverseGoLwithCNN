from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import utils.metrics as mt
from utils.models import Regressor
import torch
import sys

def get_paths_results(file_name:str, parent_folder:str, seeds:list[int]|int, shape:tuple[int,int], delta:int):
    """
    Gets the paths to the specified file for each seed, based on the provided parameters.
    The function assumes that paths were generated using the build_dir_tree function and that the 
    file_name contains either 'test', 'train', or 'val' to determine the correct subdirectory for .npz files, 
    and that .csv files are located in the Results subdirectory.

    Args:
        file_name (str): The name of the file to find (should include 'test', 'train', or 'val' for .npz files).
        parent_folder (str): The parent directory where the data is stored.
        seeds (list[int] | int): A list of seed values to construct the paths for.
        shape (tuple[int,int]): The shape of the data, used to construct the directory names.
        delta (int): The delta value used in the directory names.

    Returns:
        dict[int:str]: A list of paths to the specified file for each seed.

    Raises:
        ValueError: If the file_name does not contain 'test', 'train', or 'val' for .npz files, or if the file extension is not recognized.
    """
    if not isinstance(seeds, list):
        seeds = [seeds]

    H,W = shape
    paths = {}
    for seed in seeds:
        shape_delta_seed_dir = Path(parent_folder)/f'{H}x{W}_{delta}_{seed}'
        
        if file_name.endswith('.csv') or file_name.endswith('.pt'):
            path2append = shape_delta_seed_dir/file_name
            if path2append.exists():
                paths[seed] = path2append
            else:
                print(f"⚠️ Warning: {path2append} does not exist.")
        else:
            raise ValueError(f"File name {file_name} does not have a recognized extension (.npz, .csv or .pt).")

    return paths

def display_results(files_paths:dict[int, Path], distinguisher:str, cols2omit:int = None, num_rows:int = None,  
                    display_plot:bool=True, print_results:bool=False):
    """
    Displays and/or prints the results from multiple CSV files, grouping metrics by a distinguishing substring and 
    computing mean and standard deviation. This function assumes that the metrics in each CSV file are organized 
    such that they can be grouped into  pairs (e.g., train/validation for each metric).

    Args:
        files_paths (dict[int, Path]): Dictionary mapping seeds to paths of the CSV files to process.
        distinguisher (str): Substring used to distinguish between the two groups of metrics (e.g., 'val' for validation metrics).
        cols2omit (int, optional): Number of columns to omit from the end of each DataFrame. Defaults to None.
        num_rows (int, optional): Number of rows for the subplot grid. Required if display_plot is True. Defaults to None.
        display_plot (bool, optional): Whether to display the results as plots. If False, only prints results. Defaults to True.
        print_results (bool, optional): Whether to print the results in addition to plotting. Defaults to False.

    Raises:
        AssertionError: If no file paths are provided, if the columns cannot be paired, or if the grid distribution is invalid.

    """
    assert len(files_paths) > 0, "No CSV paths provided."   

    # Dict to save the data:
    g1_metrics = {}
    g2_metrics = {}

    # Iterate over files:
    for path in files_paths.values():
        # Read csv
        df = pd.read_csv(path)
        if cols2omit:
            df = df.drop(columns=df.columns[-cols2omit:])
        
        # Assert if the remaining metrics can be paired.
        assert len(df.columns) % 2 == 0, (f"Impossible to pair the metrics, check the columns of '{path}'.\n"
                                f"Columns after dropping {cols2omit} columns: {len(df.columns)}.")
        
        for m in df.columns:

            if not distinguisher in m:
                g1_metrics.setdefault(m,[]).append(df[m].values)

            if distinguisher in m:
                g2_metrics.setdefault(m,[]).append(df[m].values)

    # Compute mean and std values:
    g1_metrics_mean = {m: np.mean(v, axis=0) for m, v in g1_metrics.items()}
    g1_metrics_std  = {m: np.std(v,  axis=0) for m, v in g1_metrics.items()}

    g2_metrics_mean = {m: np.mean(v, axis=0) for m, v in g2_metrics.items()}
    g2_metrics_std  = {m: np.std(v,  axis=0) for m, v in g2_metrics.items()}

    # Plots:
    
    if not display_plot:
        print('--- Metrics computed during execution ---')
        for g1_m, g2_m in zip(g1_metrics_mean, g2_metrics_mean):
            print(f'{g1_m}: {g1_metrics_mean[g1_m][-1]:.4f} ± {g1_metrics_std[g1_m][-1]:.4f}')
            print(f'{g2_m}: {g2_metrics_mean[g2_m][-1]:.4f} ± {g2_metrics_std[g2_m][-1]:.4f}')
            print('='*50)
    else:
        if print_results:
            print('--- Metrics computed during execution ---')
            for g1_m, g2_m in zip(g1_metrics_mean, g2_metrics_mean):
                print(f'{g1_m}: {g1_metrics_mean[g1_m][-1]:.4f} ± {g1_metrics_std[g1_m][-1]:.4f}')
                print(f'{g2_m}: {g2_metrics_mean[g2_m][-1]:.4f} ± {g2_metrics_std[g2_m][-1]:.4f}')
                print('='*50)
        # Assert if the pairs can be distributed in the plots:
        num_pairs = len(df.columns) // 2
        assert num_pairs % num_rows == 0, f"Invalid grid distribution: {num_pairs} metric pairs cannot be arranged in {num_rows} row(s).\n"

        num_cols = num_pairs // num_rows

        _, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 4 * num_rows))
        axes = np.array(axes).flatten()  
        epochs = np.arange(1, len(next(iter(g1_metrics_mean.values()))) + 1)

        for ax, (g1_m, g2_m) in zip(axes, zip(g1_metrics_mean, g2_metrics_mean)):

            # Train
            ax.plot(epochs, g1_metrics_mean[g1_m], color='blue', label=f'{g1_m}: {np.mean(g1_metrics_mean[g1_m][-1]):.4f} ± {np.mean(g1_metrics_std[g1_m][-1]):.4f}')
            ax.fill_between(epochs,
                            g1_metrics_mean[g1_m] - g1_metrics_std[g1_m],
                            g1_metrics_mean[g1_m] + g1_metrics_std[g1_m],
                            alpha=0.5)

            # Validation
            ax.plot(epochs, g2_metrics_mean[g2_m], color='red', label=f'{g2_m}: {np.mean(g2_metrics_mean[g2_m][-1]):.4f} ± {np.mean(g2_metrics_std[g2_m][-1]):.4f}')
            ax.fill_between(epochs,
                            g2_metrics_mean[g2_m] - g2_metrics_std[g2_m],
                            g2_metrics_mean[g2_m] + g2_metrics_std[g2_m],
                            alpha=0.5)

            ax.set_title(g1_m)
            ax.set_xlabel('Epoch')
            ax.legend(loc='best')
            ax.grid(True)

        plt.tight_layout()
        plt.show()
    
def display_test_results(files_paths:dict[int, Path], cols2omit:int = None):
    """
    Displays and/or prints the results from multiple CSV files, grouping metrics by a distinguishing substring and 
    computing mean and standard deviation. This function assumes that the metrics in each CSV file are organized 
    such that they can be grouped into  pairs (e.g., train/validation for each metric).

    Args:
        files_paths (dict[int, Path]): Dictionary mapping seeds to paths of the CSV files to process.
        distinguisher (str): Substring used to distinguish between the two groups of metrics (e.g., 'val' for validation metrics).
        cols2omit (int, optional): Number of columns to omit from the end of each DataFrame. Defaults to None.
        num_rows (int, optional): Number of rows for the subplot grid. Required if display_plot is True. Defaults to None.
        display_plot (bool, optional): Whether to display the results as plots. If False, only prints results. Defaults to True.
        print_results (bool, optional): Whether to print the results in addition to plotting. Defaults to False.

    Raises:
        AssertionError: If no file paths are provided, if the columns cannot be paired, or if the grid distribution is invalid.

    """
    assert len(files_paths) > 0, "No CSV paths provided."   

    # Dict to save the data:
    g1_metrics = {}

    # Iterate over files:
    for path in files_paths.values():
        # Read csv
        df = pd.read_csv(path)
        if cols2omit:
            df = df.drop(columns=df.columns[-cols2omit:])
    
        for m in df.columns:

            g1_metrics.setdefault(m,[]).append(df[m].values)

    # Compute mean and std values:
    g1_metrics_mean = {m: np.mean(v, axis=0) for m, v in g1_metrics.items()}
    g1_metrics_std  = {m: np.std(v,  axis=0) for m, v in g1_metrics.items()}

    print('--- Metrics computed during execution ---')
    for g1_m in g1_metrics_mean:
        print(f'{g1_m}: {g1_metrics_mean[g1_m][-1]:.4f} ± {g1_metrics_std[g1_m][-1]:.4f}')
        
    print('='*50)
             
def computeCM(paths_cm:dict[int, Path], paths_thrs:dict[int,Path], metrics2compute:list[str], shape:tuple[int,int]):
    """Compute confusion matrix metrics from prediction and ground truth data.
    
    Loads CSV files containing initial state predictions and ground truth values,
    applies thresholds, and computes specified metrics across all seeds.
    
    Args:
        paths_cm: Dictionary mapping seed indices to CSV file paths containing
            predictions and ground truth states.
        paths_thrs: Dictionary mapping seed indices to CSV file paths containing
            threshold predictions.
        metrics2compute: List of metric names to compute (e.g., 'accuracy', 'precision').
        shape: Tuple of (rows, cols) representing the board dimensions.
    
    Returns:
        None. Prints computed metrics with mean values and standard deviations.
    """
    
    # Load the data:
    cols2select = [f'start_{i}' for i in range(shape[0] * shape[1])]
    cols2select_gt = [f'gt_{i}' for i in range(shape[0]*shape[1])]
    title = f'--- Results Initial states ({len(paths_cm)} seeds) ---'
 
    y_true = []
    y_pred = []
    thresholds = []
    for path_cm, path_thrs in zip(paths_cm.values(), paths_thrs.values()):
        
        thrs = pd.read_csv(path_thrs)['threshold_pred'].values
        true_states = pd.read_csv(path_cm)[cols2select_gt].values
        pred_states = pd.read_csv(path_cm)[cols2select].values
 
        y_true.append(true_states)
        y_pred.append(pred_states)
        thresholds.append(thrs)

    mean_values, stds = mt.computeCM_bxb(y_pred, y_true, thresholds, metrics2compute)
    
    print(title)
    for metric_name, mean_value, std_value in zip(metrics2compute, mean_values, stds):
        print(f'{metric_name}: {mean_value:.4f} ± {std_value:.4f}')

   
def display_states(path2tabs: Path, paths2model: dict[int, Path], shape: tuple[int, int], model_hp: dict):

    H, W = shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print('❌ GPU is not detected, interrupting execution...')
        sys.exit(1)

    # Get the seed from path2tabs = <name>_seed.csv
    seed = int(path2tabs.stem.split('_')[-1])      # Path no es subscriptable: usar .stem
    path2model = paths2model[seed]                 # seed debe ser int para casar con dict[int, Path]

    # Headers
    gtin, init_cols, gtfin, stop_cols = [], [], [], []
    for i in range(int(np.prod(shape))):
        gtin.append(f'gtin_{i}')
        gtfin.append(f'gtfin_{i}')                 # corregido: era 'gftin_' (typo)
        init_cols.append(f'start_{i}')
        stop_cols.append(f'stop_{i}')

    # Load the data
    df_tabs = pd.read_csv(path2tabs, sep=',')
    init_pred = df_tabs[init_cols].values          # heatmap inicial (continuo) -> "Heatmap"
    init_true = df_tabs[gtin].values               # estado inicial real (binario) -> "Initial State"
    fin_pred  = df_tabs[stop_cols].values          # evolución DiffGoL (continua) -> "DiffGoL Evolution"
    fin_true  = df_tabs[gtfin].values              # estado final real (binario) -> "Final State"

    # Load the model
    model = torch.compile(Regressor(n_hidden_filters=model_hp['n_hidden_filters'], kernel_size=model_hp['kernel_size'],
                                    mlp_hidden=model_hp['mlp_hidden'], use_stats=model_hp['use_stats'],
                                    padding_mode=model_hp['padding_mode']).to(device))
    checkpoint = torch.load(path2model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Un threshold por tablero, inferido del heatmap inicial
    model.eval()
    with torch.no_grad():
        init_pred_torch = torch.tensor(init_pred, dtype=torch.float32).reshape(-1, 1, H, W).to(device)
        thresholds = model(init_pred_torch).cpu().numpy().reshape(-1)   # shape (N,)

    # ---- helpers -------------------------------------------------------
    def _style(ax):
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
        ax.grid(which='minor', color='gray', linewidth=0.5)
        ax.tick_params(which='both', length=0)
        ax.set_axisbelow(False)                    # rejilla por encima del imshow

    def _comparison_rgb(pred_bin, true_bin):
        pred_bin = pred_bin.astype(bool)
        true_bin = true_bin.astype(bool)
        rgb = np.zeros((H, W, 3))
        rgb[ pred_bin &  true_bin] = (0/255,   200/255, 0/255)   # TP verde
        rgb[ pred_bin & ~true_bin] = (220/255, 0/255,   0/255)   # FP rojo
        rgb[~pred_bin &  true_bin] = (0/255,   80/255,  220/255) # FN azul
        # ~pred & ~true -> TN se queda negro
        return rgb

    # ---- figura --------------------------------------------------------
    n_samples = min(9, len(df_tabs))               # primeros 9 tableros del CSV
    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    subfigs = fig.subfigures(2, 3, wspace=0.07, hspace=0.07)
    fig.suptitle('Regressor Model', fontsize=14)
    titles = ['Initial State', 'Heatmap', 'Binarized Heatmap',
              'Final State', 'DiffGoL Evolution', 'Binarized Evolution']

    for panel, subfig in enumerate(subfigs.flat):
        subfig.suptitle(titles[panel], fontsize=14)
        axs = subfig.subplots(nrows=3, ncols=3)
        for k, ax in enumerate(axs.flat):
            _style(ax)
            if k >= n_samples:
                ax.set_visible(False)
                continue

            thr = thresholds[k]
            if panel == 0:                          # Initial State (real)
                ax.imshow(init_true[k].reshape(H, W), cmap='gray', vmin=0, vmax=1)
            elif panel == 1:                        # Heatmap (predicción soft)
                ax.imshow(init_pred[k].reshape(H, W), cmap='gray', vmin=0, vmax=1)
            elif panel == 2:                        # Binarized Heatmap vs inicial real
                pred_bin = init_pred[k].reshape(H, W) > thr
                ax.imshow(_comparison_rgb(pred_bin, init_true[k].reshape(H, W)))
            elif panel == 3:                        # Final State (real)
                ax.imshow(fin_true[k].reshape(H, W), cmap='gray', vmin=0, vmax=1)
            elif panel == 4:                        # DiffGoL Evolution (soft)
                ax.imshow(fin_pred[k].reshape(H, W), cmap='gray', vmin=0, vmax=1)
            else:                                   # Binarized Evolution vs final real
                pred_bin = fin_pred[k].reshape(H, W) > thr
                ax.imshow(_comparison_rgb(pred_bin, fin_true[k].reshape(H, W)))

    # Legend
    legend_elements = [
        mpatches.Patch(color=(1, 1, 1),                 label="Live (true)", ec="gray"),
        mpatches.Patch(color=(0, 0, 0),                 label="Dead (true)", ec="gray"),
        mpatches.Patch(color=(0/255,   200/255, 0/255), label="TP"),
        mpatches.Patch(color=(220/255, 0/255,   0/255), label="FP"),
        mpatches.Patch(color=(0/255,   80/255,  220/255), label="FN"),
        mpatches.Patch(color=(0, 0, 0),                 label="TN"),
    ]
    fig.legend(handles=legend_elements, loc="center left", ncol=1, fontsize=10,
               framealpha=0.8, bbox_to_anchor=(1, 0.5))
    plt.show()

def computeOtsu(paths2cm:dict[int, Path], shape:tuple):
    """Compute Otsu thresholds for the predicted states of each seed.

    Args:
        paths2cm (dict[int, Path]): Mapping from seed identifier to the CSV
            path containing the predicted states.
        shape (tuple): Board shape used to reshape each prediction into a
            2D grid before computing thresholds.

    Returns:
        dict[int, numpy.ndarray]: Dictionary mapping each seed to the Otsu
        threshold(s) computed for its predicted boards.
    """

    # Column names to take
    headers = [f'start_{i}' for i in range(0, np.prod(shape))]

    # Loop 
    thresholds_per_seed = {}
    for seed, path2cm in paths2cm.items():

        # Load predicted states
        pred_states = pd.read_csv(path2cm, sep=',')[headers].values
        # Compute otsu for each board
        thr = mt.otsu_per_board(pred_states, shape)
        # Save
        thresholds_per_seed[seed] = thr

    return thresholds_per_seed

def computeCM_Otsu(paths2cm:dict[int,Path], thresholds_otsu:dict[int, list], shape:tuple[int, int], metrics2compute:list[str]):

    """Compute confusion-matrix based metrics using per-seed Otsu thresholds.

    This function reads predicted and ground-truth board states from CSV files
    for each provided seed, applies the corresponding per-seed Otsu
    thresholds, computes requested confusion-matrix-derived metrics using
    mt.computeCM_bxb, and prints mean and standard deviation for each metric.

    Args:
        paths2cm (dict[int, Path]): Mapping from seed to the CSV Path that
            contains predicted ('start_*') and ground-truth ('gt_*') columns.
        thresholds_otsu (dict[int, list]): Mapping from seed to the list/array
            of Otsu thresholds to apply to that seed's predicted boards.
        shape (tuple[int, int]): Board shape (rows, cols) used to infer the
            number of 'start_*'/'gt_*' columns expected in the CSVs.
        metrics2compute (list[str]): List of metric names to compute (passed
            through to mt.computeCM_bxb).

    Returns:
        None: Results are printed to stdout. The underlying mt.computeCM_bxb
        return values (mean and std arrays) are not returned but are printed.

    Raises:
        KeyError: If a seed in paths2cm is missing in thresholds_otsu.
        FileNotFoundError / pandas errors: If CSVs cannot be read or do not
            contain the expected columns.
    """

    # Headers
    init_cols, gtin_cols = [],[]
    for i in range(0, np.prod(shape)):
        init_cols.append(f'start_{i}')
        gtin_cols.append(f'gt_{i}')

    # Load data
    y_true, y_pred, thr_list = [],[],[]
    for seed, path2cm in paths2cm.items():
        y_true.append(pd.read_csv(path2cm, sep=',')[gtin_cols].values)
        y_pred.append(pd.read_csv(path2cm, sep=',')[init_cols].values)
        thr_list.append(thresholds_otsu[seed])
        

    # COmpute confusion matrix metrics 
    mean_values, stds = mt.computeCM_bxb(y_pred, y_true, thr_list, metrics2compute)
    # Print results 
    print(f'--- Results Initial states ({len(paths2cm)} seeds) ---')
    for metric_name, mean_value, std_value in zip(metrics2compute, mean_values, stds):
        print(f'{metric_name}: {mean_value:.4f} ± {std_value:.4f}')
    

def computeCM_Gauss(paths2cm:dict[int,Path], shape:tuple[int,int], metrics2compute:list[str]):
    """Compute confusion-matrix metrics after Gaussian binarization.

    This function reads the initial and ground-truth states from each CSV in
    ``paths2cm``, applies Gaussian binarization to the predicted states, and
    then computes the requested confusion-matrix metrics.

    Args:
        paths2cm: Mapping from seed index to the CSV path containing the state
            data.
        shape: Board shape used to reshape flattened state vectors before
            Gaussian binarization.
        metrics2compute: List of metric names to compute with
            ``mt.computeCM``.

    Returns:
        None. The computed metrics are printed to stdout.

    Raises:
        FileNotFoundError: If a CSV file cannot be read.
        pandas.errors: If the CSVs do not contain the expected columns.
    """

    # Headers
    init_cols, gtin_cols = [],[]
    for i in range(0, np.prod(shape)):
        init_cols.append(f'start_{i}')
        gtin_cols.append(f'gt_{i}')

    # Load data
    y_true, y_pred = [], []
    for _, path2cm in paths2cm.items():
        df = pd.read_csv(path2cm, sep=',')
        y_true.append(df[gtin_cols].values)                       # gt de ESTE seed

        seed_pred   = df[init_cols].values                        # (n_boards, n_cells) de ESTE seed
        bin_boards  = mt.gauss_per_board(seed_pred, shape=shape)   # (n_boards, 15, 15) uint8
        pred_states = bin_boards.reshape(bin_boards.shape[0], -1)  # (n_boards, n_cells)
        y_pred.append(pred_states)                                # binarizado de ESTE seed

    # Compute metrics
    mean_values, stds = mt.computeCM(y_pred, y_true, metrics2compute)
    print(f'--- Results Initial states ({len(paths2cm)} seeds) ---')
    for metric_name, mean_value, std_value in zip(metrics2compute, mean_values, stds):
        print(f'{metric_name}: {mean_value:.4f} ± {std_value:.4f}')

def computeCM_direct(paths2cm:dict[int,Path],shape:tuple[int,int], metrics2compute:list[str]):


    # Headers
    init_cols, gtin_cols = [],[]
    for i in range(0, np.prod(shape)):
        init_cols.append(f'start_{i}')
        gtin_cols.append(f'gt_{i}')

    # Load data
    y_true, y_pred = [],[]
    for _, path2cm in paths2cm.items():
        df = pd.read_csv(path2cm, sep=',')
        y_true.append(df[gtin_cols].values)
        y_pred.append(df[init_cols].values)
    
    # Compute metrics
    mean_values, stds = mt.computeCM(y_pred, y_true, metrics2compute)
    print(f'--- Results Initial states ({len(paths2cm)} seeds) ---')
    for metric_name, mean_value, std_value in zip(metrics2compute, mean_values, stds):
        print(f'{metric_name}: {mean_value:.4f} ± {std_value:.4f}')
