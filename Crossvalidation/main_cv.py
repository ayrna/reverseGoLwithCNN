import os
import sys
import shutil
import random 
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sacred import Experiment
from keras import mixed_precision
import keras_tuner as kt
from keras.callbacks import TensorBoard
import Auxiliares.Funciones.Modelo.func_builder_cv as builder_cv
import Auxiliares.Funciones.Boards.func_load_npz as loader
import json

ex = Experiment('CrossValidation')

@ex.config
def config():
    seed = 1
    delta = 1
    shape = (15,15)
    path_datos = './Resultados/CV'
    epochs = 6
    hp_config = {
    # Parámetros fijos (no se buscan, se pasan tal cual)
    'kernel_size': (3, 3),      
    'threshold': 0.5,           # Umbral para las métricas
    'learning_rate':0.0001,
    'gamma': 2,
    'alpha': 0.75,
    # Rangos de Búsqueda (Hyperparameters)
    'n_hidden_filters': {
        'min': 2**4,
        'max': 2**8,
        'step':2**4
    },
    'n_hidden_convs': {
        'min': 1,
        'max': 6,
        'step': 1
    }
    }
    val_split = 0.2
    batch_size = 2**8

@ex.main
def main(seed, delta, shape, path_datos, epochs, hp_config, val_split, batch_size, _config):
# ---------------- Paso 1: Preparación de entorno ---------------- #
    
# ---------------- Paso 1.1: Inicialización de semillas números aleatorios ---------------- #
    os.environ['PYTHONHASHSEED'] = str(seed)
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,9"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism() 

# ---------------- Paso 1.2: Comprobación detección GPU ---------------- #
    physical_devices = tf.config.list_physical_devices('GPU')
    if not physical_devices:
        print('Error: no se han detectado GPUs')
        sys.exit(1)

# ---------------- Paso 1.3: Definición de paths ---------------- #
    H,W = shape
    path_datos = Path(path_datos)

    # Carpeta "./Resultados/CV/Boards/"
    dir_boards = path_datos/'Boards'
    file_train = dir_boards/f'datos_delta{delta}_{H}x{W}_train.npz'
    file_test = dir_boards/f'datos_delta{delta}_{H}x{W}_test.npz'
    
    dir_cv = path_datos/f'CrossValidacion'
    dir_cv2delete = dir_cv/f'cv_{seed}'
    dir_hps = path_datos/f'Best_Hps'
    file_hps = dir_hps/f'hps_{seed}.json'

    # Carpeta "./Resultados/CV/Train"
    dir_train = path_datos/'Train'

    dir_history = dir_train/'Historiales'
    file_history = dir_history/f'history_{seed}.csv'

    dir_modelos = dir_train/'Modelos'
    file_model = dir_modelos/ f"model_{seed}.keras"

    dir_tensorboard_global = dir_train/'Tensorboard'
    dir_tensorboard = dir_tensorboard_global/f'Tensorboard_{seed}'
    
    # Carpeta "./Resultados/CV/Test"
    dir_test = path_datos/'Test'

    dir_metricas = dir_test/'Metricas'
    file_test_results = dir_metricas/f"test_results_{seed}.csv"

    dir_predicciones = dir_test/'Predicciones'
    file_predicciones = dir_predicciones/f'predicciones_{seed}.csv'

# ---------------- Paso 1.4: Limpieza de directorios ---------------- #
    dirs2remove = [dir_tensorboard, dir_cv2delete]
    for dir in dirs2remove:
        try:
            # Elimino "./Resultados/CV/Boards/Train/Train_seed"
            shutil.rmtree(dir, ignore_errors=True)
        except Exception as e:
            print('Exception:', e)

    files2remove = [file_history, file_model, file_test_results, file_predicciones, file_hps]

    for file in files2remove:
        try:
            os.remove(file)
        except Exception as e:
            print('Exception:', e)


    # Creamos directorios
    dirs2create = [dir_history, dir_modelos, dir_tensorboard, dir_metricas, dir_predicciones, dir_hps]

    for path in dirs2create:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print('Exception:', e)

# ----------------------- Paso 1: CV --------------------------------- #
    def build_model(hp):
            return builder_cv.modelo_clasico(hp, hp_config=hp_config, shape=shape)
    
    tb_callback = TensorBoard(
        log_dir=dir_tensorboard,
        histogram_freq=1)
    tuner = kt.RandomSearch(
        hypermodel=build_model,                 
        objective=kt.Objective("val_loss", direction="min"), 
        max_trials=40,                          
        seed=seed,                                
        directory=dir_cv,    
        project_name=f'cv_{seed}',                                  
        executions_per_trial=1                
    )

    # Cargamos dataset train
    _ ,init_train, fin_train = loader.load_npz(file_train, f'datos_delta{delta}_{H}x{W}_train.npz')

    # Construcción de target: [iniciales, finales]
    X = fin_train.reshape((-1, shape[0], shape[1], 1)) # (k, M, M, 1)
    Y = init_train.reshape((-1, shape[0], shape[1], 1))  
    del  init_train, fin_train

    # Shuffle:
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    
    # Split train - val:
    num_val = int(len(X) * val_split)
    X_val, Y_val = X[:num_val], Y[:num_val]
    X_train, Y_train = X[num_val:], Y[num_val:]


    tuner.search(
        X_train, Y_train,
        epochs=epochs, 
        validation_data = (X_val, Y_val),
        callbacks=[tb_callback],
        verbose=1
        )
    
    best_model = tuner.get_best_models(1)[0]
    best_hps = tuner.get_best_hyperparameters()[0]

    with open(file_hps, 'w') as f:
        json.dump(best_hps.values, f, indent=4)

    best_model.save(file_model)
    
    # ----------------------- TEST DEL MODELO ----------------------- #

    # Cargar dataset
    _ ,iniciales_test, finales_test = loader.load_npz(file_test, f'datos_delta{delta}_{H}x{W}_test.npz')

    X_test = finales_test.reshape((-1, shape[0], shape[1], 1))
    Y_test = iniciales_test.reshape((-1, shape[0], shape[1], 1))

    
    del iniciales_test, finales_test 

    # Evaluación
    test_results = best_model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1, return_dict=True)
    
    # Crear DataFrame predicciones
    df_results = pd.DataFrame([test_results])
    df_results.to_csv(file_test_results, index=False)

    # Predicciones
    predicciones = best_model.predict(X_test, batch_size=batch_size, verbose=1)

    iniciales_predichos = predicciones[..., 0]
    
    celulas = shape[0]*shape[1]
    iniciales_predichos = iniciales_predichos.reshape(-1, celulas)
    
    init_cols = [f'start_{i}' for i in range(celulas)]

    full_data = iniciales_predichos
    full_cols =  init_cols

    # Crear DataFrame predicciones
    df_predicciones = pd.DataFrame(full_data, columns=full_cols)
    df_predicciones.to_csv(file_predicciones, index=False)

if __name__=='__main__':
    ex.run_commandline()