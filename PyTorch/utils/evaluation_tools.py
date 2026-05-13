from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils.plotting import multiGrays, multiComparision
from utils.data import load_npz
import utils.metrics as mt

def get_paths_results(file_name:str, parent_folder:str, seeds:list[int]|int, shape:tuple[int,int], delta:int, model_name:str=None):
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
        model_name (str, optional): The name of the model used.

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
        
        if '.npz' in file_name:
            if 'test' in file_name:
                if model_name not in ['UNet', 'CNN']:
                    path2append = shape_delta_seed_dir/f'ConwayStates/Test/{file_name}'
                else: 
                    path2append = shape_delta_seed_dir/f'ConwayStates/Test/{model_name}/{file_name}'
                if path2append.exists():
                    paths[seed] = path2append
                else:
                    print(f"⚠️ Warning: {path2append} does not exist.")
            elif 'train' in file_name:
                path2append = shape_delta_seed_dir/f'ConwayStates/Train/{model_name}/{file_name}'
                if path2append.exists():
                    paths[seed] = path2append
                else:
                    print(f"⚠️ Warning: {path2append} does not exist.")
            elif 'val' in file_name:
                path2append = shape_delta_seed_dir/f'ConwayStates/Val/{model_name}/{file_name}'
                if path2append.exists():
                    paths[seed] = path2append
                else:
                    print(f"⚠️ Warning: {path2append} does not exist.")
            else:
                raise ValueError(f"File name {file_name} does not contain 'test', 'train', or 'val' to determine the correct subdirectory.")
        elif ('.csv' in file_name) or ('.pt' in file_name):
            path2append = shape_delta_seed_dir/f'Results/{model_name}/{file_name}'
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

    
def display_states(file_pred:str, file_test:str, shape:tuple,threshold:float, titles:list[str], 
                   subplots_grid:tuple, model_name:str, figsize:tuple, random:bool=False):

    """
    Plots a 2×3 grid of subfigures comparing true and predicted boards for a
    Conway's Game of Life experiment.

    Layout:
        ┌─────────────────┬──────────────────┬───────────────────────────┐
        │  Initial · True │  Initial · Heat  │  Initial · Comparison     │
        ├─────────────────┼──────────────────┼───────────────────────────┤
        │  Final   · True │  Final   · Heat  │  Final   · Comparison     │
        └─────────────────┴──────────────────┴───────────────────────────┘

    Columns 1–2 are rendered in grayscale via `multiGrays`.
    Column 3 is a cell-wise TP/FP/FN/TN colour map via `multiComparison`.
    A shared legend is placed to the right of the figure.

    Args:
        file_pred (str): Path to the CSV file containing predicted states.
        file_test (str): Path to the NPZ file containing true states.
        shape (tuple): Board dimensions (height, width).
        threshold (float): Probability threshold for binarising predictions.
        titles (list[str]): Exactly 6 subfigure titles in row-major order.
        subplots_grid (tuple): Inner grid layout (nrows, ncols) per subfigure.
        model_name (str): Figure super-title. Pass None to omit it.
        figsize (tuple): Overall figure size (width, height) in inches.
        random (bool): If True, selects a random subset of boards shared
            across all subfigures. Defaults to False.

    Returns:
        None: Displays the figure.
    
    Raises:
        AssertionError if the number of titles is not 6.
    """

    assert len(titles) == 6, f'You must provide exactly 6 titles. You provided {len(titles)}.'

    stop_cols = [f"stop_{i}" for i in range(shape[0]*shape[1])]
    init_cols = [f'start_{i}' for i in range(shape[0]*shape[1])]

    # Load predicted states:
    df_preds = pd.read_csv(file_pred)

    init_pred_prob = df_preds[init_cols].values
    init_pred_bin = init_pred_prob > threshold
    fin_pred_prob = df_preds[stop_cols].values
    fin_pred_bin = fin_pred_prob > threshold

    # Load true states:
    _, init_true, fin_true = load_npz(file_test, 'test.npz')

    # Tabs2plot:
    tabs2plot = [init_true, init_pred_prob, init_pred_bin,
                 fin_true,  fin_pred_prob,  fin_pred_bin]

    # Pre-filter with random indices so multiGrays/multiComparison iterate sequentially:
    rows, cols = subplots_grid
    n_samples = len(tabs2plot[0])
    n_plots_per_sub = rows * cols

    if random:
        rng = np.random.default_rng()
        limit = min(n_samples, n_plots_per_sub)
        indices = rng.choice(n_samples, size=limit, replace=False)
        tabs2plot = [t[indices] for t in tabs2plot]
        init_true = init_true[indices]
        fin_true = fin_true[indices]

    # Figure:
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    subfigs = fig.subfigures(2, 3, wspace=0.07, hspace=0.07)

    if model_name is not None:
        fig.suptitle(model_name, fontsize=14)

    for i, subfig in enumerate(subfigs.flat):
        subfig.suptitle(titles[i], fontsize=14)
        current_axs = subfig.subplots(nrows=rows, ncols=cols)

        if i == 2:    # Initial binary → comparison vs init_true
            multiComparision(tabs2plot[i], init_true, subplots_grid, shape, current_axs)
        elif i == 5:  # Final binary → comparison vs fin_true
            multiComparision(tabs2plot[i], fin_true,  subplots_grid, shape, current_axs)
        else:         # Grayscale (true boards or heatmaps)
            multiGrays(tabs2plot[i], subplots_grid, shape, current_axs)

    # Legend for comparison subfigures:
    legend_elements = [
        mpatches.Patch(color=(1, 1, 1),                   label="Live (true)",  ec="gray"),
        mpatches.Patch(color=(0, 0, 0),                   label="Dead (true)",  ec="gray"),
        mpatches.Patch(color=(0/255,   200/255, 0/255),   label="TP"),
        mpatches.Patch(color=(220/255, 0/255,   0/255),   label="FP"),
        mpatches.Patch(color=(0/255,   80/255,  220/255), label="FN"),
        mpatches.Patch(color=(0,       0,       0),       label="TN"),
    ]
    fig.legend(
    handles=legend_elements, loc="center left", ncol=1, fontsize=10, framealpha=0.8, bbox_to_anchor=(1, 0.5))   
    plt.show()

def compute_metrics(paths_pred:dict[int,Path], paths_test:dict[int,Path], shape:tuple, state:str, th_range:list[tuple], threshold:float, max_penalty:float= 20,
                    display_plot:bool=True, bins:int =30, colors:list[str] =['blue', 'red'], figsize:tuple=(10,6),alpha:float=0.5,log_scale:bool=True):
    
    list2check = ['Initial', 'Final', 'initial', 'final', 'init', 'fin']
    assert state in list2check, f'"{state}" is not a valid state. Valid states: {list2check}'

    if state in ['Initial', 'initial', 'init']:
        cols2select = [f'start_{i}' for i in range(shape[0]*shape[1])]
        init = True
        title = f'Initial States ({len(paths_pred)} seeds)'
        
    elif state in ['Final', 'final', 'fin']:
        cols2select = [f"stop_{i}" for i in range(shape[0]*shape[1])]
        init = False
        title = f'Final States ({len(paths_pred)} seeds)'
    
    y_true = []
    y_pred = []
    for path_pred, path_true in zip(paths_pred.values(), paths_test.values()):

        if init:
            _, true_states, _ = load_npz(path_true, 'test.npz')
        else:
            _, _, true_states = load_npz(path_true,'test.npz')
        
        pred_states = pd.read_csv(path_pred)[cols2select].values

        y_true.append(true_states)
        y_pred.append(pred_states)
    
    print(f'--- Results {title} ---')
    mt.density(y_pred, y_true)
    print("=" * 50)
    for i in range(len(th_range)):
        low_th, high_th = th_range[i]
        mt.huc(y_pred, low_th, high_th)
    mt.fuzziness_index(y_pred, shape)
    print("=" * 50)
    mt.mse(y_pred, y_true, shape, threshold)
    print("=" * 50)
    mt.wasserstein_optim(y_pred, y_true, shape, max_penalty)

    # Plot:
    if display_plot:
        mt.plot_density(y_pred, y_true, bins, colors, figsize, alpha, title, log_scale)



