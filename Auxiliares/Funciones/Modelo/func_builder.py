import tensorflow as tf
from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization 
from keras import backend as K
from keras.optimizers import Adam
import Auxiliares.Clases.class_Metricas as mt
import Auxiliares.Clases.class_GoLayer as GoLayer

### ------------------------------------------------------------------------------ ###
        #########           Modelo con Capa Sharp Conway 2D         ######### 
        #########                  FBCE + BCE + Reg                 #########
### ------------------------------------------------------------------------------ ###
def modelo_complejo(hp_config, shape=(20,20), delta = 1):
    """
    Builds and compiles a classical Convolutional Neural Network (CNN) with differentiable Game of Life Layer.

    This function defines a fully convolutional architecture aimed at predicting 
    initial Game of Life states from final states, and their differentiable Game of Life final states. 

    Args:
        hp_config (dict)
        shape (tuple[int, int], optional): Spatial dimensions (height, width) of 
            the input boards. Defaults to (20, 20).
        delta (int): number of generations between the initial and final states.

    Returns:
        keras.Model: A compiled Keras model initialized with the sampled hyperparameters, 
            custom loss, and evaluation metrics.
    """
    # --------------------------------- Parámetros Fijos ------------------------------------- #
    # Arquitectura del modelo
    kernel_size = hp_config['kernel_size']
    n_hidden_convs = hp_config['n_hidden_convs']
    n_hidden_filters = hp_config['n_hidden_filters']
    threshold = hp_config['threshold']
    learning_rate = hp_config['learning_rate']
    # Loss:
    lambda_phys = hp_config['lambda_phys']
    lambda_bin = hp_config['lambda_bin']
    gamma = hp_config['gamma']
    alpha = hp_config['alpha']
    # Capa GoL:
    epsilon = hp_config['epsilon']
    order = hp_config['order']
    
    # ----------------------------------- MODELO ----------------------------------- #
    M = shape[0]
    N = shape[1]
    
    input_final = Input(shape=(M, N, 1), name='input_final')

    x = Conv2D(n_hidden_filters, kernel_size, padding='same', activation='relu')(input_final)
    x = BatchNormalization()(x)

    for _ in range(n_hidden_convs):
        x = Conv2D(n_hidden_filters, kernel_size, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)

    # Predicciones:
    predicted_init = Conv2D(1, kernel_size, padding='same', activation='sigmoid')(x)
    predicted_final = GoLayer.ConwayLayer(delta=delta, epsilon=epsilon, order=order)(predicted_init)

    # Agrupación de predicciones
    output_total = tf.keras.layers.Concatenate(axis = -1, name='output_total')([predicted_init,predicted_final]) 

    model = Model(inputs=input_final, outputs=output_total, name='diffGoL')

    # Algoritmo de optimización Adam
    optimizer = Adam(learning_rate = learning_rate)
    model.compile(
        optimizer=optimizer,
        loss = mt.customLoss(lambda_phys, lambda_bin, gamma, alpha), # FBCE_init + BCE_fin + Reg_init
        metrics= [mt.Accuracy(threshold=threshold, wl=True), 
                    mt.AccuracyFin(threshold=threshold), 
                    mt.Recall(threshold=threshold, wl=True),
                    mt.Recall_fin(threshold=threshold,wl=True),
                    mt.Specificity(threshold=threshold, wl=True),
                    mt.Specificity_fin(threshold=threshold, wl=True),
                    mt.Precision(threshold=threshold, wl=True),
                    mt.Precision_fin(threshold=threshold, wl=True),
                    mt.F1Score(threshold=threshold, wl=True),
                    mt.Absolute(threshold=threshold, wl=True),
                    mt.ComponenteFBCE(gamma=gamma, alpha=alpha),
                    mt.ComponenteBCE(lambda_phys=lambda_phys),
                    mt.ComponenteBinarization(lambda_bin=lambda_bin)
                    ])
        
    return model

### ------------------------------------------------------------------------------ ###
           #########           Modelo Clásico        ######### 
        #########                FBCE + Reg             #########
### ------------------------------------------------------------------------------ ###
def modelo_clasico(hp_config, shape=(20,20)):
    """
    Builds and compiles a classical Convolutional Neural Network (CNN).

    This function defines a fully convolutional architecture aimed at predicting 
    initial Game of Life states from final states.

    Args:
        hp_config (dict)
        shape (tuple[int, int], optional): Spatial dimensions (height, width) of 
            the input boards. Defaults to (20, 20).

    Returns:
        keras.Model: A compiled Keras model initialized with the sampled hyperparameters, 
            custom loss, and evaluation metrics.
    """
    # --------------------------------- Parámetros Fijos ------------------------------------- #
    # Arquitectura del modelo
    kernel_size = hp_config['kernel_size']
    n_hidden_convs = hp_config['n_hidden_convs']
    n_hidden_filters = hp_config['n_hidden_filters']
    threshold = hp_config['threshold']
    learning_rate = hp_config['learning_rate']
    # Loss:
    gamma = hp_config['gamma']
    alpha = hp_config['alpha']
    
    
    # ----------------------------------- MODELO ----------------------------------- #
    M = shape[0]
    N = shape[1]
    
    input_final = Input(shape=(M, N, 1), name='input_final')

    x = Conv2D(n_hidden_filters, kernel_size, padding='same', activation='relu')(input_final)
    x = BatchNormalization()(x)

    for _ in range(n_hidden_convs):
        x = Conv2D(n_hidden_filters, kernel_size, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)

    # Predicciones:
    predicted_init = Conv2D(1, kernel_size, padding='same', activation='sigmoid')(x)
    
    model = Model(inputs=input_final, outputs=predicted_init, name='SharpConway')

    # Algoritmo de optimización Adam
    optimizer = Adam(learning_rate = learning_rate)
    model.compile(
        optimizer=optimizer,
        loss = mt.classic_customLoss(gamma, alpha), # FBCE_init 
        metrics= [mt.Accuracy(threshold=threshold, wl=False), 
                    mt.Recall(threshold=threshold, wl=False),
                    mt.Specificity(threshold=threshold, wl=False),
                    mt.Precision(threshold=threshold, wl=False),
                    mt.F1Score(threshold=threshold, wl=False),
                    mt.Absolute(threshold=threshold, wl=False)
                    ])
        
    return model


### ------------------------------------------------------------------------------ ###
        #########           Modelo con Capa Sharp Conway 2D         ######### 
        #########                  FBCE + BCE + Reg                 #########
### ------------------------------------------------------------------------------ ###
def modelo_complejo_dinamico(hp_config, shape=(20,20), delta = 1):
    """
    Builds and compiles a classical Convolutional Neural Network (CNN) with differentiable Game of Life Layer.
    It allows dynamic regularization.

    This function defines a fully convolutional architecture aimed at predicting 
    initial Game of Life states from final states, and their differentiable Game of Life final states. 

    Args:
        hp_config (dict)
        shape (tuple[int, int], optional): Spatial dimensions (height, width) of 
            the input boards. Defaults to (20, 20).
        delta (int): number of generations between the initial and final states.

    Returns:
        keras.Model: A compiled Keras model initialized with the sampled hyperparameters, 
            custom loss, and evaluation metrics.
    """
    # --------------------------------- Parámetros Fijos ------------------------------------- #
    # Arquitectura del modelo
    kernel_size = hp_config['kernel_size']
    n_hidden_convs = hp_config['n_hidden_convs']
    n_hidden_filters = hp_config['n_hidden_filters']
    threshold = hp_config['threshold']
    learning_rate = hp_config['learning_rate']
    # Loss:
    lambda_phys = tf.Variable(hp_config['lambda_phys'], trainable=False, dtype=tf.float32, name='lambda_phys')
    lambda_bin = tf.Variable(hp_config['lambda_bin'], trainable=False, dtype=tf.float32, name='lambda_bin')
    gamma = hp_config['gamma']
    alpha = hp_config['alpha']
    # Capa GoL:
    epsilon = hp_config['epsilon']
    order = hp_config['order']
    
    # ----------------------------------- MODELO ----------------------------------- #
    M = shape[0]
    N = shape[1]
    
    input_final = Input(shape=(M, N, 1), name='input_final')

    x = Conv2D(n_hidden_filters, kernel_size, padding='same', activation='relu')(input_final)
    x = BatchNormalization()(x)

    for _ in range(n_hidden_convs):
        x = Conv2D(n_hidden_filters, kernel_size, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)

    # Predicciones:
    predicted_init = Conv2D(1, kernel_size, padding='same', activation='sigmoid')(x)
    predicted_final = GoLayer.ConwayLayer(delta=delta, epsilon=epsilon, order=order)(predicted_init)

    # Agrupación de predicciones
    output_total = tf.keras.layers.Concatenate(axis = -1, name='output_total')([predicted_init,predicted_final]) 

    model = Model(inputs=input_final, outputs=output_total, name='Dynamic_diffGoL')

    # Algoritmo de optimización Adam
    optimizer = Adam(learning_rate = learning_rate)
    model.compile(
        optimizer=optimizer,
        loss = mt.customLoss(lambda_phys, lambda_bin, gamma, alpha), # FBCE_init + BCE_fin + Reg_init
        metrics= [mt.Accuracy(threshold=threshold, wl=True), 
                    mt.AccuracyFin(threshold=threshold), 
                    mt.Recall(threshold=threshold, wl=True),
                    mt.Recall_fin(threshold=threshold,wl=True),
                    mt.Specificity(threshold=threshold, wl=True),
                    mt.Specificity_fin(threshold=threshold, wl=True),
                    mt.Precision(threshold=threshold, wl=True),
                    mt.Precision_fin(threshold=threshold, wl=True),
                    mt.F1Score(threshold=threshold, wl=True),
                    mt.Absolute(threshold=threshold, wl=True),
                    mt.ComponenteFBCE(gamma=gamma, alpha=alpha),
                    mt.ComponenteBCE(lambda_phys=lambda_phys),
                    mt.ComponenteBinarization(lambda_bin=lambda_bin)
                    ])
    model.lambda_phys = lambda_phys
    model.lambda_bin = lambda_bin
    return model