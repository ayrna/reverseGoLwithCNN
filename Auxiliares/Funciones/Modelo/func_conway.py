import numpy as np
from scipy.signal import convolve2d

def game_of_life(tablero_inicial, delta=1):
    """
    Executes Conway's Game of Life using 2D convolution (Vectorized).

    Args:
        tablero_inicial (numpy.ndarray): Binary matrix (0s and 1s) representing the initial state.
        steps (int): Number of generations to simulate.

    Returns:
        numpy.ndarray: The final state of the board after the specified steps.
    """
    # 1. Definimos el Kernel (Filtro)
    # Una matriz 3x3 de unos, con un 0 en el centro.
    # Al convolucionar, esto suma los 8 vecinos y descarta la propia célula.
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    # Trabajamos sobre una copia para no mutar el input original (Best Practice)
    tablero = tablero_inicial.astype(int) 

    for _ in range(delta):
        # 2. Conteo de vecinos 
        # mode='same': Mantiene el tamaño original del tablero.
        # boundary='fill': Asume 0s fuera de los bordes (o 'wrap' para mundo toroidal).
        vecinos = convolve2d(tablero, kernel, mode='same', boundary='wrap')

        # 3. Aplicación de Reglas (Lógica Booleana Pura)
        # Regla Supervivencia 1.0 y Nacimiento: Nace o Sobrevive si tiene 3 vecinos.
        # Regla Supervivencia 2.0: Sobrevive si tiene 2 vecinos Y ya estaba viva.
        # Todo lo demás se convierte en False (0) automáticamente.
        tablero = ((vecinos == 3) | ((tablero == 1) & (vecinos == 2))).astype(int)

    return tablero