import numpy as np
import matplotlib.pyplot as plt


#####################################################################################################

def Gray(tableros, tipo, ncols, nrows, figsize, shape):

    """
    Plots boards in white (live cell) and black (dead cell).

    Args:
        boards ( array): Array containing the boards to plot. Does not need to be reshaped beforehand.
        board_type (str): Type of the board. Expected values: 'base', 'initial', or 'final'.
        ncols (int): Number of columns for the figure subplots. Must be a divisor of len(boards).
        figsize (tuple[int, int]): Size of the figure window.
        board_shape (tuple[int, int]): Dimensions of the individual boards (height, width).

    Returns:
        None: Displays a subplot figure with the requested boards. 
        
    """

    # ----- Paso 1: Reshape de tableros a (mxn) -----
    
    tableros = tableros.reshape(-1, shape[0], shape[1])

    # ----- Paso 2: Características de la gráfica -----
    tableros2plot = ncols*nrows
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.suptitle(f'Muestra aleatoria de {tableros2plot} tableros {tipo}', y = 0.94)
    ax = ax.flatten()
    for i in range(tableros2plot):
        ax[i].imshow(tableros[i], cmap="gray")
        ax[i].set_xticks([])
        ax[i].set_yticks([])
   
    

#####################################################################################################

def multiGray(tableros, ncols, nrows, shape, axs, random=False, forced_indices=None): 
    """
    Renders a grid of Conway's Game of Life boards in grayscale (white=live, black=dead).

    Supports synchronized plotting across multiple figures by accepting pre-computed 
    indices (`forced_indices`), or local random/sequential selection. Automatically 
    handles grids larger than the number of provided boards.

    Args:
        tableros (array): Array of boards to plot. Can be flat; will be reshaped.
        ncols (int): Number of columns in the subplot grid.
        nrows (int): Number of rows in the subplot grid.
        shape (tuple[int, int]): Target dimensions (height, width) for reshaping each board.
        axs (array): Array of matplotlib Axes objects to draw on.
        random (bool): If True, selects boards randomly. Ignored if 
            `forced_indices` is provided. Defaults to False.
        forced_indices (array): Specific indices to plot. Useful for 
            synchronizing the view across different sets of boards (e.g., initial states 
            vs. model predictions). Defaults to None.

    Returns:
        None: Displays the plot.
    """
    
    tabs = tableros.reshape(-1, shape[0], shape[1])
    n_total = len(tabs)
    n_plots = ncols * nrows
    ax_flat = np.array(axs).flatten()
    limit = min(n_total, n_plots)
    
    # Lógica de Selección de Índices
    if forced_indices is not None:
        # CASO 1: Índices inyectados desde fuera (Sincronizados)
        indices = forced_indices[:limit] # Aseguramos no pasarnos del límite de plots
    elif random:
        # CASO 2: Aleatorio local (Desincronizado / Solo una figura)
        rng = np.random.default_rng()
        indices = rng.choice(n_total, size=limit, replace=False)
    else:
        # CASO 3: Secuencial
        indices = np.arange(limit)
    
    # Renderizado
    for i in range(n_plots):
        if i < len(indices):
            idx = indices[i]
            # Protección extra por si el índice externo es mayor que el array actual
            if idx < len(tabs): 
                ax_flat[i].imshow(tabs[idx], cmap="gray")
            
        ax_flat[i].set_xticks([])
        ax_flat[i].set_yticks([])
    
    
    
    

#####################################################################################################
def Densidades(tableros, ncols, nrows, figsize, escalado=300):
    """
    Plots a 2D spatial grid representing the density of live cells for each board.
    
    Calculates the mean density of each board and projects them onto a 2D meshgrid 
    defined by `ncols` and `nrows`. The point size and color scale with the density.

    Args:
        tableros ( array): Array of boards. Shape should allow flattening to 
            (ncols * nrows) boards.
        ncols (int): Number of columns in the 2D projection grid.
        nrows (int): Number of rows in the 2D projection grid.
        figsize (tuple[int, int]): Size of the figure window.
        escalado (int, optional): Scaling factor for the scatter point sizes. Defaults to 300.

    Returns:
        None: Displays the plot.
    """
    # ----- Paso 1: Cálculo de densidades y de máximo de densidad -----
    densidades = np.mean(tableros, axis=1)
    matrix_densities = densidades.reshape(ncols, nrows)
    max_densidad = np.max(matrix_densities)

    # ----- Paso 2: Creación de malla de puntos -----
    rows, cols = matrix_densities.shape
    x_indices, y_indices = np.meshgrid(np.arange(cols), np.arange(rows))

    # ----- Paso 3: Puntos de tableros con sus densidades -----
    x = x_indices.flatten()
    y = y_indices.flatten()
    d = matrix_densities.flatten()

    # ----- Paso 4: Gráfico de puntos con escalado según densidades -----
    plt.figure(figsize=figsize)
    scatter = plt.scatter(x, y, s=d * escalado, c=d, cmap='plasma', 
                        edgecolors='black', linewidth=0.5, alpha=0.9)

    # ----- Paso 5: Para que coincida con el plot de la función Gris() -----
    plt.gca().invert_yaxis()  # Origen (0,0) arriba a la izquierda
    plt.colorbar(scatter, label='Densidad por tablero')
    title= f'Distribución de densidades: {len(tableros)} tableros \n Máximo {max_densidad:.3f}'
    plt.title(title)
    plt.grid(True, linestyle=':', alpha=0.4)

    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    

#####################################################################################################

def ScatterDensitiesfromTab(tableros, figsize, escalado=50, tipo="aleatorios"):
    """
    Creates a 1D scatter plot of board densities directly from the board arrays.

    Plots the density of each board against its index in the array. Point sizes 
    and colors are proportional to the density value.

    Args:
        tableros ( array): Array of boards from which densities are calculated.
        figsize (tuple[int, int]): Size of the figure window.
        escalado (int, optional): Scaling factor for the scatter point sizes. Defaults to 50.
        tipo (str, optional): Descriptive string for the plot title (e.g., 'random', 
            'initial'). Defaults to "aleatorios".

    Returns:
        None: Displays the plot.
    """

    # ----- Paso 1: Cálculo de densidades y de máximo de densidad -----
    densidades = np.mean(tableros, axis=1)
    max_densidad = np.max(densidades)
    # ----- Paso 2: Eje OX -----
    x = np.arange(1, len(tableros) + 1)
    
    # ----- Paso 3: Figura -----
    plt.figure(figsize=figsize)
    scatter = plt.scatter(x, densidades, s=densidades*escalado, c=densidades, cmap="plasma", 
                edgecolors="black", linewidth=0.05, alpha=0.9)
    plt.colorbar(scatter, label='Densidad por tablero')
    title= f'Distribución de densidades: {len(tableros)} tableros {tipo}\n Máximo {max_densidad:.3f}'
    plt.title(title)
    plt.title(title)
    plt.xlabel("Índice del Tablero")
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.xticks([])
    # plt.ylim([0,1])
    plt.yticks([])
    plt.tight_layout()
    

#####################################################################################################
def ScatterDensities(densidades, figsize=(10, 6)):
    """
    Creates a 1D scatter plot of pre-calculated board densities.

    Plots the provided density values against their index. Unlike `ScatterDensitiesfromTab`,
    this function expects a 1D array of floats, avoiding recalculation.

    Args:
        densidades ( array): 1D array of pre-calculated density floats.
        figsize (tuple[int, int], optional): Size of the figure window. Defaults to (10, 6).

    Returns:
        None: Displays the plot.
    """
    
    # Eje X: Índices de los tableros (0 a 200,000)
    x = np.arange(len(densidades))
    max_densidad = np.max(densidades)
    plt.figure(figsize=figsize)
    
    plt.scatter(x, densidades, s=10, c=densidades, cmap="viridis", alpha=0.5)
    
    plt.colorbar(label='Densidad por tablero')
    plt.title(f'Densidades para {len(densidades)} muestras \n Máximo: {max_densidad:.3f}')
    plt.xlabel("Índice del Tablero")
    plt.grid(True, alpha=0.3)
    
    plt.yticks([])
    plt.tight_layout()
    

#####################################################################################################
def Histograma(densidades, figsize=(12,5),tipo="aleatorios"):
    """
    Plots a histogram showing the frequency distribution of board densities.

    Args:
        densidades ( array): 1D array of pre-calculated density floats.
        figsize (tuple[int, int], optional): Size of the figure window. Defaults to (12, 5).
        tipo (str, optional): Descriptive string for the plot title. Defaults to "aleatorios".

    Returns:
        None: Displays the plot.
    """
    fig = plt.figure(figsize=(12, 5))
    plt.hist(densidades, bins="auto", color='blue', edgecolor='black', alpha=0.7)
    maximo = np.max(densidades)
    plt.title(f"Distribución de densidades de {len((densidades))} tableros {tipo}\n Máximo de {maximo:.3f}")
    plt.xlabel("Densidad células vivas")
    plt.ylabel("Frecuencia")
    plt.grid(axis='y', alpha=0.3)
    


#####################################################################################################
def Comparativa(tabs_predichos,tabs_reales, delta, shape, nfilas, ncols, tabs='finales', figsize=(10,10)):
    """
    Plots a visual comparison grid between predicted and ground truth boards.

    Renders a multiclass overlap image where colors indicate classification outcomes:
    - Green: True Positive (Model correctly predicted alive).
    - Black: True Negative (Model correctly predicted dead).
    - Red: False Positive (Model predicted alive, but actually dead).
    - Blue: False Negative (Model predicted dead, but actually alive).

    Args:
        tabs_predichos ( array): Array of predicted boards.
        tabs_reales ( array): Array of ground truth boards.
        delta (int/float): The time step or generation gap (used in title).
        shape (tuple[int, int]): Dimensions (height, width) for reshaping each board.
        nfilas (int): Number of rows in the subplot grid.
        ncols (int): Number of columns in the subplot grid.
        tabs (str, optional): Description of the board states for the title. Defaults to 'finales'.
        figsize (tuple[int, int], optional): Size of the figure window. Defaults to (10, 10).

    Returns:
        None: Displays the comparative subplot grid.
    """

    posibilidades = len(tabs_predichos)
    rng = np.random.default_rng(seed = 43)
    indices = rng.choice(posibilidades, size=nfilas*ncols, replace=False)
    print(indices[0])
    # Seleccionar filas específicas
    predichos = tabs_predichos[indices]
    reales = tabs_reales[indices]
    
    fig, axes = plt.subplots(nfilas, ncols, figsize=figsize)
    fig.suptitle(f'Comparativa Tableros {tabs} para \n ({delta}, {shape[0]}, {shape[1]})')
    axes = axes.flatten()

    i = 0
    for final, y_real in zip(reales, predichos):
         y_real = y_real.reshape(shape[0],shape[1])
         y_pred = final.reshape(shape[0],shape[1])

         comparativa = np.zeros((shape[0],shape[1],3))
        # Criterio de colores:
     
         comparativa[(y_real == 1) & (y_pred == 1)] = [0,1,0] # Si coinciden, verde (TP)
         comparativa[(y_real == 0) & (y_pred == 0)] = [0,0,0]  # Si coinciden, negro (TN)
         comparativa[(y_real == 0) & (y_pred == 1)] = [1,0,0]  # Muerto pero vivo, rojo (FP)
         comparativa[(y_real == 1) & (y_pred == 0)] = [0,0,1]# Vivo pero muerto, azul (FN)
         axes[i].imshow(comparativa)
         axes[i].set_xticks([])
         axes[i].set_yticks([])
        #  axes[i].set_title(f'Tablero {indices[i]}')

         # Añadir mallado
         axes[i].set_xticks(np.arange(-0.5, shape[0], 1), minor=True)
         axes[i].set_yticks(np.arange(-0.5, shape[1], 1), minor=True)
         axes[i].grid(which='minor', color='white', linestyle='-', linewidth=0.4)

         # Eliminar totalmente las etiquetas de los ticks (sin numeritos, sin marcas)
         axes[i].tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
          # Quitar los ticks mayores y menores (solo deja la grid)
         axes[i].set_xticks([])
         axes[i].set_yticks([])

         i +=1
   
