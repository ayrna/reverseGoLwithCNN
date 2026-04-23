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

import Auxiliares.Funciones.Modelo.func_builder as builder
import Auxiliares.Funciones.Boards.func_load_npz as loader
import Auxiliares.Funciones.Boards.func_board_generator as dynamic

ex = Experiment('Train_GoL')

@ex.config
def config():
    seed = 1
    delta = 1
    shape = (15,15)
    path_datos = './Resultados/Resultados_GoL'
    epochs = 6
    hold = True
    epochs2hold = 2
    dict_hps = {
                'kernel_size':3,
                'n_hidden_convs':6,
                'n_hidden_filters':256,
                'threshold': 0.5,
                'learning_rate': 0.0001,
                'lambda_phys': 0.5,
                'lambda_bin':1,
                'gamma': 2,
                'alpha': 0.75,
                'epsilon': 50,
                'order': 4
                }
    batch_size = 2**8


@ex.main
def main(seed, delta, shape, path_datos, epochs, hold, epochs2hold, dict_hps, batch_size, _config):
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
    path_datos = Path(path_datos) # ./Resultados/Resultads_{Modelo}

    # Carpeta "./Resultados/Resultados_{MODELO}/Boards/Train"
    dir_boards_train = path_datos/'Boards/Train'
    dir_boards_train_seed= dir_boards_train/f'Train_{seed}' #

    # Carpeta "./Resultados//Resultados_{MODELO}/Boards/Val"
    dir_boards_val = path_datos/'Boards/Val'
    dir_boards_val_seed = dir_boards_val/f'Val_{seed}' # 

    # Carpeta "./Resultados/Resultados_{MODELO}/Boards/Test"
    dir_boards_test = path_datos/'Boards/Test'
    file_test_seed = dir_boards_test/f'test_{seed}.npz'
    
    # Carpeta "./Resultados/Resultados_{MODELO}/Train"
    dir_train = path_datos/'Train'

    dir_history = dir_train/'Historiales'
    file_history = dir_history/f'history_{seed}.csv'

    dir_modelos = dir_train/'Modelos'
    file_model = dir_modelos/ f"model_{seed}.keras"

    dir_tensorboard = dir_train/'Tensorboard'
    file_tensorboard = dir_tensorboard/f"tensorboard_{seed}.csv"


# ---------------- Paso 1.4: Limpieza de directorios ---------------- #
    dirs2remove = [dir_boards_train_seed, dir_boards_val_seed]
    for directorio in dirs2remove:
        try:
            # Elimino "./Resultados/Boards/Train/Train_seed"
            shutil.rmtree(directorio, ignore_errors=True)
        except Exception as e:
            print('Exception:', e)

    files2remove = [file_history, file_model, file_tensorboard, file_test_seed]

    for file in files2remove:
        try:
            os.remove(file)
        except Exception as e:
            print('Exception:', e)


    # Creamos directorios
    dirs2create = [dir_boards_train_seed, dir_boards_val_seed, dir_history, dir_modelos, dir_tensorboard, dir_boards_test]
            
    for path in dirs2create:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print('Exception:', e)

# ----------------------- Paso 2: Entrenamiento ----------------------- #

# ---------------- Paso 2.0: Inicialización modelo y creación del validation dataset ---------------- #
    model = builder.modelo_complejo(dict_hps, shape, delta)

    # Variables para control manual de los callbacks
    best_val_loss = float('inf')
    history_global = [] # Lista para acumular diccionarios
    patience_cnt = 0    # Contador para LR manual 
    patience_dts = 0
    for epoch in range(epochs): # [0, 999]
# ---------------- Paso 2.1: Creación y carga de dataset dinámico (k, H*W) ---------------- #
        file_boards_train = dir_boards_train_seed/f'train_epoch_{epoch}.npz'
        file_boards_val = dir_boards_val_seed/f'val_epoch_{epoch}.npz'
        
        if epoch % (epochs2hold -1) == 0:
            change_epoch = True
            patience_dts = 0
        else:
            change_epoch = False

        if (hold==True):
            if (change_epoch or change_patience): 
                model.optimizer.learning_rate.assign(dict_hps['learning_rate']) # Reinicio el learning rate
                dynamic.board_generator(seed = seed, 
                                        delta=delta, 
                                        shape=shape, 
                                        dir_npz=[dir_boards_train_seed, dir_boards_val_seed], 
                                        name=[f'train_epoch_{epoch}', f'val_epoch_{epoch}']
                                        )  # TrainVal=True, val_split=0.2, n_objetivo = 60k num_tableros = 120k, densidad_min= 0.05, densidad_max = 0.95, batch_size=5_000)
                # Cargar dataset
                _ ,iniciales_train, finales_train = loader.load_npz(file_boards_train, f'train_epoch_{epoch}.npz')
                _ ,iniciales_val, finales_val = loader.load_npz(file_boards_val, f'val_epoch_{epoch}.npz')
                            
        else: 
            model.optimizer.learning_rate.assign(dict_hps['learning_rate'])
            dynamic.board_generator(seed = seed, 
                                        delta=delta, 
                                        shape=shape, 
                                        dir_npz=[dir_boards_train_seed, dir_boards_val_seed], 
                                        name=[f'train_epoch_{epoch}', f'val_epoch_{epoch}']
                                        )  #  TrainVal=True, val_split=0.2, n_objetivo = 60k num_tableros = 120k, densidad_min= 0.05, densidad_max = 0.95, batch_size=5_000)
            # Cargar dataset
            _ ,iniciales_train, finales_train = loader.load_npz(file_boards_train, f'train_epoch_{epoch}.npz')
            _ ,iniciales_val, finales_val = loader.load_npz(file_boards_val, f'val_epoch_{epoch}.npz')  
        
# ---------------- Paso 2.2: Transformación del dataset para TF  ---------------- #
        # Construcción de train_dataset:
        X_train = finales_train.reshape((-1, shape[0], shape[1], 1)) # (k, M, M, 1)
        Y_train = np.concatenate([iniciales_train.reshape((-1, shape[0], shape[1], 1)), X_train], axis=-1) # (k, M,M, 2)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        train_dataset = train_dataset.shuffle(buffer_size=50000, seed=seed, reshuffle_each_iteration=True)
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Construcción de val_dataset:
        X_val = finales_val.reshape((-1, shape[0], shape[1], 1)) # (k, M, M, 1)
        Y_val = np.concatenate([iniciales_val.reshape((-1, shape[0], shape[1], 1)), X_val], axis=-1) # (k, M,M, 2)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
        val_dataset = val_dataset.shuffle(buffer_size=50000, seed=seed, reshuffle_each_iteration=True)
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ---------------- Paso 2.3: Fit de 1 epoch ---------------- #
        history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=1,
                verbose=1
            )
        
        # Guardamos las métricas
        metrics = {k: v[-1] for k, v in history.history.items()}

# ---------------- Paso 2.4: Callbacks ---------------- #
        current_val_loss = metrics['val_loss']
        # CheckPoint
        if current_val_loss < best_val_loss:
            print(f"Mejora detectada ({best_val_loss:.4f} -> {current_val_loss:.4f}). Guardando modelo...")
            best_val_loss = current_val_loss
            model.save(file_model)
            patience_cnt = 0 # Reiniciar contador si mejora
            patience_dts = 0
        else:
            patience_cnt += 1
            patience_dts += 1
            print(f"No mejora. Best: {best_val_loss:.4f}. Patience LR: {patience_cnt}. Patience Dts: {patience_dts}")

        # Reduce LR Manual 
        if patience_cnt >= 20:
            old_lr = float(model.optimizer.learning_rate.numpy()) # .numpy() si es TF tensor
            new_lr = old_lr * 0.5
            if new_lr > 1e-6:
                model.optimizer.learning_rate.assign(new_lr)
                print(f"Reducing Learning Rate: {old_lr:.6f} -> {new_lr:.6f}")
                patience_cnt = 0 # Reinicio de contador
        else:
            new_lr = float(model.optimizer.learning_rate.numpy())
        
        metrics['learning_rate'] = new_lr

        # Checheko para cambio de dataset:
        if patience_dts >=25:
            patience_dts = 0
            patience_cnt = 0
            change_patience = True
        else:
            change_patience= False
       
        # Guardo si se ha cambiado el dataset:
        if change_patience or change_epoch:
            metrics['NewDataset'] = 1 # Se guarda
        else: 
            metrics['NewDataset'] = 0 # No se guarda

        # Guardo el historial para visualizar como tensorboard:
        history_global.append(metrics)
        df_tensorboard = pd.DataFrame(history_global)
        df_tensorboard.to_csv(file_tensorboard, index=False)

        
    # Guardar historial y modelo:   
    df_history = pd.DataFrame(history_global)
    df_history.to_csv(file_history, index=False)

    
if __name__=='__main__':
    ex.run_commandline()