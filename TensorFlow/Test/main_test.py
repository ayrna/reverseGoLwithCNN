import os
import sys
import random 
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sacred import Experiment
import Auxiliares.Funciones.Boards.func_load_npz as loader
import Auxiliares.Funciones.Boards.func_board_generator as dynamic
import Auxiliares.Clases.class_Metricas as mt
import Auxiliares.Clases.class_GoLayer as GoLayer

ex = Experiment('Test_GoL')

@ex.config
def config():
    seed = 1
    delta = 1
    shape = (15,15)
    path_datos = './Resultados'
    batch_size = 2**8

@ex.main
def main(seed, delta, shape, path_datos, batch_size, _config):
# ---------------- Paso 1: Preparación de entorno ---------------- #

# ---------------- Paso 1.1: Inicialización de semillas números aleatorios ---------------- #
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# ---------------- Paso 1.2: Comprobación detección GPU ---------------- #
    physical_devices = tf.config.list_physical_devices('GPU')
    if not physical_devices:
        print('Error: no se han detectado GPUs')
        sys.exit(1)

# ---------------- Paso 1.3: Definición de paths ---------------- #
    path_datos1 = Path(path_datos)/f'Resultados_Classic' # ./Resultados/Resultados_Classic
    path_datos2 = Path(path_datos)/f'Resultados_GoL' # ./Resultados/Resultados_GoL
    path_datos3 = Path(path_datos)/f'Resultados_AnnDiffGoL' # ./Resultados/Resultados_DynamicReg
    
    # Carpeta: ./Resultados/Resultados_{Modelo}/Boards/Test
    dir_board_test1 = path_datos1/f'Boards/Test'
    file_board_test_seed1 = dir_board_test1/f'test_{seed}.npz'

    # Carpeta "./Resultados/Resultados_{MODELO}/Train/Modelos"
    file_model1 = path_datos1/'Train/Modelos'/ f"model_{seed}.keras"

    file_model2 = path_datos2/'Train/Modelos'/ f"model_{seed}.keras"

    file_model3 = path_datos3/'Train/Modelos'/ f"model_{seed}.keras"

   
    # Carpeta "./Resultados/Resultados_{Modelo}/Test/Metricas"
    dir_test_results1 = path_datos1/'Test/Metricas'
    file_test_results1 = dir_test_results1/f'test_results_{seed}.csv'

    dir_test_results2 = path_datos2/'Test/Metricas'
    file_test_results2 = dir_test_results2/f'test_results_{seed}.csv'

    dir_test_results3 = path_datos3/'Test/Metricas'
    file_test_results3 = dir_test_results3/f'test_results_{seed}.csv'

    # Carpeta "./Resultados/Resultados_{Modelo}/Test/Predicciones"
    dir_predicciones1 = path_datos1/'Test/Predicciones'
    file_predicciones1 = dir_predicciones1/f'predicciones_{seed}.csv'

    dir_predicciones2 = path_datos2/'Test/Predicciones'
    file_predicciones2 = dir_predicciones2/f'predicciones_{seed}.csv'

    dir_predicciones3 = path_datos3/'Test/Predicciones'
    file_predicciones3 = dir_predicciones3/f'predicciones_{seed}.csv'

# ---------------- Paso 1.4: Limpieza de directorios ---------------- #
    
    files2remove = [ 
                    file_test_results1, file_test_results2, file_test_results3, 
                    file_predicciones1, file_predicciones2, file_predicciones3]

    for file in files2remove:
        try:
            os.remove(file)
        except Exception as e:
            print('Exception:', e)

    # Creamos directorios
    dirs2create = [dir_board_test1, 
                   dir_test_results1, dir_test_results2, dir_test_results3, dir_predicciones1, 
                   dir_predicciones1, dir_predicciones2, dir_predicciones3]
            
    for path in dirs2create:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print('Exception:', e)

    # ---------------------- Paso 1: Cargar dataset: -------------------- #
    
    # Cargar dataset
    _ ,iniciales_test, finales_test = loader.load_npz(file_board_test_seed1, f'test_{seed}.npz')
    
    X_test = finales_test.reshape((-1, shape[0], shape[1], 1))
    Y_1 = iniciales_test.reshape((-1, shape[0], shape[1], 1))

    Y_2 = np.concatenate([Y_1, X_test], axis=-1).astype('float32')

    # ----------------------- TEST DEL MODELO ----------------------- #

    # Cargo los tres modelos:
    best_model1 = tf.keras.models.load_model(file_model1)
    best_model2 = tf.keras.models.load_model(file_model2)
    best_model3 = tf.keras.models.load_model(file_model3)

    celulas = shape[0]*shape[1]
    stop_cols = [f"stop_{i}" for i in range(celulas)]
    init_cols = [f'start_{i}' for i in range(celulas)]

    # # -------- Evaluación MODELO CLÁSICO: ----------
    test_results1 = best_model1.evaluate(X_test, Y_1, batch_size=batch_size, verbose=1, return_dict=True)

    # Crear DataFrame predicciones
    df_results1 = pd.DataFrame([test_results1])
    df_results1.to_csv(file_test_results1, index=False)

    # Predicciones
    predicciones1 = best_model1.predict(X_test, batch_size=batch_size, verbose=1)

    iniciales_predichos1 = predicciones1[..., 0]
    
    iniciales_predichos1 = iniciales_predichos1.reshape(-1, celulas)
    
    full_data1 = iniciales_predichos1
    full_cols1 = init_cols

    # Crear DataFrame predicciones
    df_predicciones1 = pd.DataFrame(full_data1, columns=full_cols1)
    df_predicciones1.to_csv(file_predicciones1, index=False)

    # # -------- Evaluación MODELO GoL: ----------
    test_results2 = best_model2.evaluate(X_test, Y_2, batch_size=batch_size, verbose=1, return_dict=True)

    # Crear DataFrame predicciones
    df_results2 = pd.DataFrame([test_results2])
    df_results2.to_csv(file_test_results2, index=False)

    # Predicciones
    predicciones2 = best_model2.predict(X_test, batch_size=batch_size, verbose=1)

    iniciales_predichos2 = predicciones2[..., 0]
    finales_predichos2 = predicciones2[..., 1]

    celulas = shape[0]*shape[1]
    iniciales_predichos2 = iniciales_predichos2.reshape(-1, celulas)
    finales_predichos2 = finales_predichos2.reshape(-1, celulas)

    full_data2 = np.hstack((finales_predichos2, iniciales_predichos2))
    full_cols2 = stop_cols + init_cols

    # Crear DataFrame predicciones
    df_predicciones2 = pd.DataFrame(full_data2, columns=full_cols2)
    df_predicciones2.to_csv(file_predicciones2, index=False)

    # # -------- Evaluación MODELO DynamicReg: ----------
    test_results3 = best_model3.evaluate(X_test, Y_2, batch_size=batch_size, verbose=1, return_dict=True)

    # Crear DataFrame predicciones
    df_results3 = pd.DataFrame([test_results3])
    df_results3.to_csv(file_test_results3, index=False)

    # Predicciones
    predicciones3 = best_model3.predict(X_test, batch_size=batch_size, verbose=1)

    iniciales_predichos3 = predicciones3[..., 0]
    finales_predichos3 = predicciones3[..., 1]

    celulas = shape[0]*shape[1]
    iniciales_predichos3 = iniciales_predichos3.reshape(-1, celulas)
    finales_predichos3 = finales_predichos3.reshape(-1, celulas)

    full_data3 = np.hstack((finales_predichos3, iniciales_predichos3))
    full_cols3 = stop_cols + init_cols

    # Crear DataFrame predicciones
    df_predicciones3 = pd.DataFrame(full_data3, columns=full_cols3)
    df_predicciones3.to_csv(file_predicciones3, index=False)


if __name__=='__main__':
    ex.run_commandline()