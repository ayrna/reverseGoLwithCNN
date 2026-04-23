import Auxiliares.Funciones.Graficas.func_graficador as graficador
import numpy as np

def SaveHist(dir_plots, bases, iniciales, finales, tipo="PostFiltro", todos=False):
    """
    Calculates and plots density histograms for different sets of boards, saving them to disk.

    Args:
        dir_plots (string): Directory path where the plots will be saved.
        bases (array, optional): Array of base boards. Shape: (num_boards, H*W).
        iniciales (array, optional): Array of initial boards. Shape: (num_boards, H*W).
        finales (array, optional): Array of final boards. Shape: (num_boards, H*W).
        tipo (string): Suffix for the filename to identify the stage. Defaults to "PostFiltro".
        save_all (bool): If True, concatenates all provided boards and plots a combined histogram. Defaults to False.
    """
    # Los tableros deben ser de la forma (num_tableros, H*W)

    if tipo == "":
        fig_histdensebase = dir_plots/f"HistDensidades_bases.png"
        fig_histdenseiniciales = dir_plots/f"HistDensidades_iniciales.png"
        fig_histdensefinales = dir_plots/f"HistDensidades_finales.png"
        fig_histdensetodos = dir_plots/f"HistDensidades_juntos.png"
    else:
        fig_histdensebase = dir_plots/f"HistDensidades{tipo}_bases.png"
        fig_histdenseiniciales = dir_plots/f"HistDensidades{tipo}_iniciales.png"
        fig_histdensefinales = dir_plots/f"HistDensidades{tipo}_finales.png"
        fig_histdensetodos = dir_plots/f"HistDensidades{tipo}_juntos.png"
    
    if isinstance(bases, np.ndarray):
        fig = graficador.Histograma(np.mean(bases, axis=1), figsize=(10,5),tipo= "base")
        fig.savefig(fig_histdensebase)

    if isinstance(iniciales, np.ndarray):
        fig = graficador.Histograma(np.mean(iniciales,axis=1), figsize=(10,5), tipo="iniciales")
        fig.savefig(fig_histdenseiniciales)

    if isinstance(finales, np.ndarray):
        fig = graficador.Histograma(np.mean(finales,axis=1), figsize=(10,5),tipo="finales")
        fig.savefig(fig_histdensefinales)

    if todos:
        todos = np.concatenate((bases, iniciales, finales), axis=0)
        fig = graficador.Histograma(np.mean(todos,axis=1), figsize=(10,5), tipo="")
        fig.savefig(fig_histdensetodos)
    
    fig.clf()
    

def SaveDense(dir_plots, bases, iniciales, finales, tipo="PostFiltro", todos=False):
    """
    Generates scatter density plots for provided sets of boards and saves them to disk.

    Args:
        dir_plots (string): Directory path where the plots will be saved.
        bases (array, optional): Array of base boards. Shape: (num_boards, H*W).
        iniciales (array, optional): Array of initial boards. Shape: (num_boards, H*W).
        finales (array, optional): Array of final boards. Shape: (num_boards, H*W).
        tipo (str): Suffix for the filename to identify the stage. Defaults to "PostFiltro".
        save_all (bool): If True, concatenates all provided boards and plots combined densities. Defaults to False.
    """
    # Los tableros deben ser de la forma (num_tableros, H*W)

    if tipo == "":
        fig_densebase = dir_plots/f"Densidades_bases.png"
        fig_denseiniciales = dir_plots/f"Densidades_iniciales.png"
        fig_densefinales = dir_plots/f"Densidades_finales.png"
        fig_densetodos = dir_plots/f"Densidades_juntos.png"
    
    else:
        fig_densebase = dir_plots/f"Densidades{tipo}_bases.png"
        fig_denseiniciales = dir_plots/f"Densidades{tipo}_iniciales.png"
        fig_densefinales = dir_plots/f"Densidades{tipo}_finales.png"
        fig_densetodos = dir_plots/f"Densidades{tipo}_juntos.png"
    
    if isinstance(bases, np.ndarray):
        fig = graficador.ScatterDensitiesfromTab(bases, figsize=(10,5), tipo="base")
        fig.savefig(fig_densebase)

    if isinstance(iniciales, np.ndarray):
        fig = graficador.ScatterDensitiesfromTab(iniciales, figsize=(10,5),tipo="iniciales")
        fig.savefig(fig_denseiniciales)

    if isinstance(finales, np.ndarray):
        fig = graficador.ScatterDensitiesfromTab(finales, figsize=(10,5),tipo="finales")
        fig.savefig(fig_densefinales)

    if todos:
        todos = np.concatenate((bases, iniciales, finales), axis=0)
        fig = graficador.ScatterDensitiesfromTab(todos, figsize=(10,5), tipo="")
        fig.savefig(fig_densetodos)
    
    fig.clf()

def SaveGray(dir_plots, tableros, shape,tipo="base", lotesize=400, ncols = 20, num_plots=3):
    """
    Saves batches of boards as grayscale grid images.

    Args:
        dir_plots (string): Directory path where the plots will be saved.
        tableros (array): Flattened array of boards. Shape: (num_boards, H*W).
        shape (tuple[int, int]): Dimensions to reshape the boards to (height, width).
        tipo (str): Prefix for the filename. Defaults to "base".
        lotesize (int): Number of boards to include per plot. Defaults to 400.
        ncols (int): Number of columns in the plot grid. Defaults to 20.
        num_plots (int): Number of batches/images to generate. Defaults to 3.
    """
    # Los tableros deben ser de la forma (num_tableros, H*W)
    
    nrows = int(lotesize/ncols)
    i = 0
    while i < num_plots:
        start = i*lotesize
        end = (i + 1)*lotesize
        lote = tableros[start:end].reshape(-1,shape[0],shape[1])
        fig = graficador.Gray(lote,tipo, ncols, nrows, (ncols, nrows), (shape[0],shape[1]))
        fig_path = dir_plots/f"{tipo}_{i}.png"
        fig.savefig(fig_path)
        i += 1
    fig.clf()