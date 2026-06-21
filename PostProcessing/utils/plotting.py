import matplotlib.pyplot as plt
import numpy as np

def multiComparision(tableros_pred, tableros_true, subplots_grid, shape, axs):
    """
    Renders a grid of cell-wise comparison boards between binary predictions and
    ground truth for Conway's Game of Life.

    Each cell is coloured according to its classification:
        - Green  → True  Positive  (pred=1, true=1)
        - Red    → False Positive  (pred=1, true=0)
        - Blue   → False Negative  (pred=0, true=1)
        - Black  → True  Negative  (pred=0, true=0)

    Args:
        tableros_pred (array): Flat or 3-D array of binary predicted boards.
        tableros_true (array): Flat or 3-D array of binary ground-truth boards.
        subplots_grid (tuple[int, int]): Grid layout (nrows, ncols).
        shape (tuple[int, int]): Target board dimensions (height, width).
        axs (array): Array of matplotlib Axes objects to draw on.

    Returns:
        None: Displays the plot in-place.
    """
    nrows, ncols = subplots_grid
    pred = tableros_pred.reshape(-1, shape[0], shape[1]).astype(bool)
    true = tableros_true.reshape(-1, shape[0], shape[1]).astype(bool)

    n_total = len(pred)
    n_plots = ncols * nrows
    ax_flat = np.array(axs).flatten()
    limit   = min(n_total, n_plots)

    for i in range(n_plots):
        ax = ax_flat[i]
        if i < limit:
            p = pred[i]
            t = true[i]
            rgb = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
            rgb[p  & t]  = [0,200, 0]   # TP → green
            rgb[p  & ~t] = [220,0,0]   # FP → red
            rgb[~p & t]  = [0,80,220]   # FN → blue
  
            ax.imshow(rgb, interpolation="nearest")

        ax.set_xticks(np.arange(-0.5, shape[0], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, shape[1], 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.4, alpha=0.75)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(which='minor', length=0)
        
    
def multiGrays(tableros, subplots_grid, shape, axs):
    """
    Renders a grid of Conway's Game of Life boards in grayscale (white=live, black=dead).

    Args:
        tableros (array): Flat or 3-D array of boards to plot.
        subplots_grid (tuple[int, int]): Grid layout (nrows, ncols).
        shape (tuple[int, int]): Target board dimensions (height, width).
        axs (array): Array of matplotlib Axes objects to draw on.

    Returns:
        None: Displays the plot in-place.
    """
    nrows, ncols = subplots_grid
    tabs = tableros.reshape(-1, shape[0], shape[1])

    n_total = len(tabs)
    n_plots = ncols * nrows
    ax_flat = np.array(axs).flatten()
    limit   = min(n_total, n_plots)

    for i in range(n_plots):
        ax = ax_flat[i]
        if i < limit:
            ax.imshow(tabs[i], cmap="gray")

        ax.set_xticks(np.arange(-0.5, shape[0], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, shape[1], 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.4, alpha=0.75)
       
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(which='minor', length=0)