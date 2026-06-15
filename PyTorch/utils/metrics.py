import numpy as np
import matplotlib.pyplot as plt
import ot 
import cv2
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
import torch 
from torchmetrics.functional import roc, auroc
import utils.experiment_tools as tools

def fuzziness_index(y_pred, shape: tuple):
    """
    Computes the mean fuzziness index for a set of predicted boards.

    Fuzziness measures how uncertain the predictions are: 0 means fully binary
    (all cells are 0 or 1), 1 means maximum uncertainty (all cells at 0.5).

    Args:
        y_pred (array): Predicted probability values, shape (n_samples, h*w).
        shape (tuple): Board dimensions (height, width).

    Returns:
        tuple[float, float]: Mean and standard deviation of the fuzziness index.
    """
    y_pred = np.array(y_pred)
    boards_prob = y_pred.reshape(-1, shape[0], shape[1])

    sample_fuzziness = []
    for board_prob in boards_prob:
        board_prob_clipped = np.clip(np.asarray(board_prob), 0.0, 1.0)
        fuzziness = float(np.mean(4.0 * board_prob_clipped * (1.0 - board_prob_clipped)))
        sample_fuzziness.append(fuzziness)

    mean = np.mean(sample_fuzziness)
    std  = np.std(sample_fuzziness)
    print(f"Fuzziness Index: {mean:.4f} ± {std:.4f}")

def huc(y_pred, low_th:float, high_th:float):
    """
    Computes the High Uncertainty Cells percentage (HUC%) for a set of predicted boards.

    A cell is considered uncertain if its predicted probability falls strictly
    between `low_th` and `high_th`. Reports the mean and standard deviation
    across all boards.

    Args:
        y_pred (array): Predicted probability values, shape (n_samples, n_cells).
        low_th (float): Lower bound of the uncertainty interval (exclusive).
        high_th (float): Upper bound of the uncertainty interval (exclusive).

    Returns:
        None: Prints mean and std of the uncertain cell percentage.
    """
    y_pred = np.array(y_pred)
    uncertain_percentages = []
    for board in y_pred:
        mask = (board > low_th) & (board < high_th)
        percentage = np.sum(mask) / board.size
        uncertain_percentages.append(percentage)
    uncertain_percentages = np.array(uncertain_percentages)
    mean_uncertain = uncertain_percentages.mean()
    std_uncertain = uncertain_percentages.std(ddof=1)
    print(f'HUC% {low_th,high_th}: {mean_uncertain:.2%} ± {std_uncertain:.2%}')

def density(y_pred, y_true):
    """
    Compares the cell density (mean activation) between predicted and true boards.

    For each seed, computes the mean and std of per-board densities (variability
    across boards within a seed). Then reports the mean of those values across
    seeds, and the absolute error per seed.

    Args:
        y_pred (list of arrays): Predicted probability values, one array per seed,
            each of shape (n_samples, n_cells).
        y_true (list of arrays): Ground-truth binary values, one array per seed,
            each of shape (n_samples, n_cells).

    Returns:
        None: Prints per-seed densities, std across boards, error per seed, and the
        maximum predicted value.
    """
    n_seeds = len(y_true)

    mean_true, std_true = [], []
    mean_pred, std_pred = [], []
    for i in range(n_seeds):
        density_true_per_board = np.array(y_true[i]).mean(axis=1)  # shape (n_samples,)
        density_pred_per_board = np.array(y_pred[i]).mean(axis=1)  # shape (n_samples,)

        mean_true.append(density_true_per_board.mean())
        mean_pred.append(density_pred_per_board.mean())

        std_true.append(density_true_per_board.std(ddof=1))  # std ENTRE tableros
        std_pred.append(density_pred_per_board.std(ddof=1))  # std ENTRE tableros

    # Convert to numpy arrays before any arithmetic
    mean_true = np.array(mean_true)  # shape (n_seeds,)
    mean_pred = np.array(mean_pred)  # shape (n_seeds,)
    std_true  = np.array(std_true)   # shape (n_seeds,)
    std_pred  = np.array(std_pred)   # shape (n_seeds,)

    mean_density_true = mean_true.mean()
    std_density_true  = std_true.mean()

    mean_density_pred = mean_pred.mean()
    std_density_pred  = std_pred.mean()

    # Absolute error per seed, then summarise
    error_per_seed = np.abs(mean_true - mean_pred)
    mean_error = error_per_seed.mean()
    std_error  = error_per_seed.std(ddof=1)

    print(f'Density (true): {mean_density_true:.2%} ± {std_density_true:.2%}')
    print(f'Density (pred): {mean_density_pred:.2%} ± {std_density_pred:.2%}')
    print(f'Error/seed:     {mean_error:.2%} ± {std_error:.2%}')
    print(f'Max value predicted: {np.max(np.concatenate([np.array(s).flatten() for s in y_pred])):.4f}')

def plot_density(y_pred, y_true, bins:int, colors:list[str], figsize:tuple, alpha:float, title:str, log_scale:bool):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    plt.figure(figsize=figsize)
    plt.hist(y_true.flatten(), bins=bins, color=colors[0], alpha=alpha, edgecolor='black',
            label='True', density=True, range=(0, 1))

    plt.hist(y_pred.flatten(), bins=bins, color=colors[1], alpha=alpha, edgecolor='black',
            label='Pred', density=True,  range=(0, 1))

    if log_scale:
        plt.yscale('log')
        plt.ylabel('Frequency Density (Log scale)')
    else:
        plt.ylabel('Frequency Density')

    plt.title(f'Histogram {title}')
    
    plt.xlabel('Pixel value')
    plt.legend()
    plt.show()
    

def mse(y_pred, y_true, shape: tuple, threshold: float):
    """
    Computes the Mean Squared Error (MSE) for model predictions.

    Evaluates MSE for both continuous probabilities and binarized outcomes,
    comparing them against the ground truth and an all-zeros baseline.

    Args:
        y_pred (array): Predicted probability values, shape (n_samples, h*w).
        y_true (array): Ground-truth binary values, shape (n_samples, h*w).
        shape (tuple): Board dimensions (height, width).
        threshold (float): Probability threshold for binarising predictions.

    Returns:
        None: Prints the MSE metrics.
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    preds_prob = y_pred.reshape(-1, shape[0], shape[1])
    preds_bin  = preds_prob > threshold
    all_zeros  = np.zeros(shape[0] * shape[1])
    true_boards = y_true.reshape(-1, shape[0], shape[1])  

    mse_prob = []
    mse_bin  = []
    mse_zero = []

    for tab_prob, tab_bin, real in zip(preds_prob, preds_bin, true_boards):
        real_flat = real.ravel()
        prob_flat = tab_prob.ravel()
        bin_flat  = tab_bin.astype("float32").ravel()

        mse_prob.append(np.mean((real_flat - prob_flat) ** 2))
        mse_bin.append( np.mean((real_flat - bin_flat)  ** 2))
        mse_zero.append(np.mean((real_flat - all_zeros) ** 2))

    print(f"MSE (prob):{np.mean(mse_prob):.4f} ± {np.std(mse_prob):.4f}")
    print(f"MSE (bin):{np.mean(mse_bin):.4f} ± {np.std(mse_bin):.4f}")
    print(f"MSE (all-zero): {np.mean(mse_zero):.4f} ± {np.std(mse_zero):.4f}")


def wasserstein(y_pred, y_true, shape:tuple, max_penalty:float = 20.0):
    """
    Computes the mean Wasserstein (Earth Mover's) distance between predicted and
    true boards.

    Treats each 2D board as a probability mass function and computes the optimal
    transport cost to transform the predicted distribution into the true one.
    Edge cases (empty boards) are handled via a fixed penalty.

    Args:
        y_pred (array): Predicted probability values, shape (n_samples, h*w).
        y_true (array): Ground-truth binary values, shape (n_samples, h*w).
        shape (tuple): Board dimensions (height, width).
        max_penalty (float): Penalty applied when one board has mass and the
            other is empty. Defaults to 20.0.

    Returns:
        None: Prints mean and standard deviation of the Wasserstein distances.
    """

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    n_seeds = len(y_pred)
    preds = y_pred.reshape(n_seeds, -1, shape[0], shape[1])  # (n_seeds, n_samples, h, w)
    trues = y_true.reshape(n_seeds, -1, shape[0], shape[1])

    # Precompute cost matrix
    coords = np.array([[i, j] for i in range(shape[0]) for j in range(shape[1])])
    cost_matrix = cdist(coords, coords, metric="sqeuclidean")

    emds = []
    for i in range(n_seeds):
        emds_seed = []
        for tab_pred, tab_true in zip(preds[i], trues[i]):

            mass_pred = tab_pred.flatten().astype(np.float64)
            mass_true = tab_true.flatten().astype(np.float64)

            sum_pred = mass_pred.sum()
            sum_true = mass_true.sum()

            if sum_pred == 0 and sum_true == 0:
                emds_seed.append(0.0)
            elif sum_pred == 0 or sum_true == 0:
                emds_seed.append(max_penalty)
            else:
                mass_pred_norm = mass_pred / sum_pred
                mass_true_norm = mass_true / sum_true
                emd = ot.emd2(mass_pred_norm, mass_true_norm, cost_matrix)
                emds_seed.append(np.sqrt(emd))
        emds.append(np.mean(emds_seed))

    print(f"Wasserstein (EMD): {np.mean(emds):.4f} ± {np.std(emds):.4f}")

def wasserstein_optim(y_pred, y_true, shape: tuple, max_penalty: float = 20.0, n_jobs: int = -1):

    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)

    n_seeds = len(y_pred)
    n_cells = shape[0] * shape[1]
    preds = y_pred.reshape(n_seeds, -1, n_cells)
    trues = y_true.reshape(n_seeds, -1, n_cells)

    coords = np.mgrid[:shape[0], :shape[1]].reshape(2, -1).T.astype(np.float64)
    cost_matrix = cdist(coords, coords, metric="sqeuclidean")

    def _single_emd(mass_pred, mass_true):
        sum_pred = mass_pred.sum()
        sum_true = mass_true.sum()
        if sum_pred == 0 and sum_true == 0:
            return 0.0
        if sum_pred == 0 or sum_true == 0:
            return max_penalty
        return np.sqrt(ot.emd2(mass_pred / sum_pred, mass_true / sum_true,
                               cost_matrix, numThreads=1))

    emds = []
    for i in range(n_seeds):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_single_emd)(preds[i, j], trues[i, j])
            for j in range(preds.shape[1])
        )
        emds.append(np.mean(results))

    print(f"Wasserstein (EMD): {np.mean(emds):.4f} ± {np.std(emds):.4f}")

def computeROC(y_pred, y_true, mean_fpr):
    """
    Compute per-board ROC metrics, averaged within each seed.
 
    For every seed, this function iterates over its boards and, for each board,
    computes the ROC curve, the AUROC, and the threshold that maximizes Youden's
    J statistic (TPR - FPR). The per-board TPR curves are interpolated onto a
    shared FPR grid. The per-board TPR curves, AUROC values, and thresholds are
    then averaged across the boards of that seed, yielding a single curve, a
    single AUROC, and a single threshold per seed.
 
    Args:
        y_pred (array-like): Predicted scores/probabilities with shape
            ``(n_seeds, n_boards, n_cells)``.
        y_true (array-like): Binary ground-truth labels aligned with ``y_pred``,
            with the same shape.
        mean_fpr (array-like): Common FPR grid used to interpolate the per-board
            TPR curves before averaging.
 
    Returns:
        tuple[list[np.ndarray], list[float], list[float]], list[[]]: Three lists, each of
        length ``n_seeds``:
            - Mean interpolated TPR curve per seed (averaged over its boards).
            - Mean AUROC per seed (averaged over its boards).
            - Mean Youden threshold per seed (averaged over its boards).
    """
 
    # Seed loop
    tprs_interp, aucs, best_ths, all_board_ths = [], [], [], []
    for pred, gtruth in zip(y_pred, y_true):

        # Convert to tensor -->  shape (n_seeds, n_boards, n_cells)
        pred = torch.tensor(pred, dtype=torch.float32)
        gtruth = torch.tensor(gtruth, dtype=torch.bool)

        # Board loop
        board_tprs, board_aucs, board_ths = [], [], []
        for b_pred, b_gtruth in zip(pred, gtruth):

            # Compute the ROC curve and the AUROC for this board
            fpr, tpr, thresholds = roc(b_pred, b_gtruth, task="binary")
            score = auroc(b_pred, b_gtruth, task="binary")
 
            # To numpy
            fpr, tpr, thresholds = fpr.numpy(), tpr.numpy(), thresholds.numpy()
 
            # Best threshold by Youden's J statistic --> max(tpr - fpr)
            best_idx = np.argmax(tpr - fpr)
            board_ths.append(thresholds[best_idx])
 
            # Interpolate this board's curve onto the common grid
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            board_tprs.append(interp_tpr)
            board_aucs.append(score.item())

        # Average over boards -> a single value for this seed
        tprs_interp.append(np.mean(board_tprs, axis=0))
        aucs.append(np.mean(board_aucs))
        best_ths.append(np.mean(board_ths))
        all_board_ths.append(np.array(board_ths))
 
    return tprs_interp, aucs, best_ths, all_board_ths

def computeCM(y_pred, y_true, threshold, metrics2compute):

    # Loop
    per_seed = {metric: [] for metric in metrics2compute}
    for pred, gtruth in zip(y_pred, y_true):

        # Convert to tensor -->  shape (n_seeds, n_boards, n_cells)
        pred = torch.tensor(pred, dtype=torch.float32)
        gtruth = torch.tensor(gtruth, dtype=torch.long)

        # Compute each metric:
        for metric in metrics2compute:

            # Get it
            metric_fn = tools.get_metric(metric, task='binary', threshold=threshold)

            # Compute it
            value = metric_fn(pred, gtruth)        

            # Save it
            per_seed[metric].append(value.item())

    # Save
    results = np.array([per_seed[m] for m in metrics2compute])

    # Compute mean values
    mean_values = results.mean(axis=1)
    stds = results.std(axis=1)  

    return mean_values, stds

def computeCM_bxb(y_pred, y_true, thresholds, metrics2compute):

    # Loop
    per_seed = {metric: [] for metric in metrics2compute}
    for pred, gtruth, ths in zip(y_pred, y_true, thresholds):

        # Convert to tensor -->  shape (n_seeds, n_boards, n_cells)
        pred = torch.tensor(pred, dtype=torch.float32)
        gtruth = torch.tensor(gtruth, dtype=torch.long)

        assert len(ths) == pred.shape[0], f"Thresholds ({len(ths)}) != n_boards ({pred.shape[0]})"
        
        # Compute each metric:
        for metric in metrics2compute:

            # Get it
            per_board = []
            for b_pred, b_gtruth, threshold in zip(pred, gtruth, ths):
                metric_fn = tools.get_metric(metric, task='binary', threshold=float(threshold))

                # Compute it
                per_board.append(metric_fn(b_pred, b_gtruth).item())

            # Save it
            per_seed[metric].append(np.array(per_board).mean())

    # Save
    results = np.array([per_seed[m] for m in metrics2compute])

    # Compute mean values
    mean_values = results.mean(axis=1)
    stds = results.std(axis=1)  

    return mean_values, stds
    
def otsu_per_board(y_pred, shape=(15, 15)):
    
    boards = np.asarray(y_pred, dtype=np.float64).reshape(-1, *shape)
    thresholds = np.empty(boards.shape[0])
    binarized = np.empty_like(boards, dtype=np.uint8)

    for i, board in enumerate(boards):
        img = np.round(board * 255).astype(np.uint8)
        if img.min() == img.max():            # tablero constante: Otsu no aplica
            thresholds[i] = 0.5               # fallback documentado
            binarized[i] = (board >= 0.5).astype(np.uint8)
            continue
        t, board_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresholds[i] = t / 255.0
        binarized[i] = (board_bin // 255).astype(np.uint8)

    return thresholds, binarized  
        
