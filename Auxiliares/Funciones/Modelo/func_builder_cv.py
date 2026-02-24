from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization 
from keras.optimizers import Adam
import Auxiliares.Clases.class_Metricas as mt

def modelo_clasico(hp, hp_config, shape=(20,20)):
    """
    Builds and compiles a classical Convolutional Neural Network (CNN) for hyperparameter tuning.

    This function defines a fully convolutional architecture aimed at predicting 
    initial Game of Life states from final states. It utilizes Keras Tuner to explore 
    different network depths (`n_hidden_convs`) and widths (`n_hidden_filters`).

    Args:
        hp (keras_tuner.HyperParameters): Object used to define and track the 
            hyperparameter search space during tuning.
        hp_config (dict): Configuration dictionary containing:
            - Fixed parameters: 'kernel_size', 'threshold', 'learning_rate', 
              'alpha_loss', 'gamma_loss'.
            - Search space boundaries (dicts with 'min', 'max', 'step' keys) for:
              'n_hidden_filters', 'n_hidden_convs'.
        shape (tuple[int, int], optional): Spatial dimensions (height, width) of 
            the input boards. Defaults to (20, 20).

    Returns:
        keras.Model: A compiled Keras model initialized with the sampled hyperparameters, 
            custom FBCE loss, and evaluation metrics, ready for the tuning loop.
    """
    # --------------------------------- Parámetros Fijos ------------------------------------- #
    kernel_size = hp_config['kernel_size']
    threshold = hp_config['threshold']
    
    learning_rate = hp_config['learning_rate']
    alpha_loss = hp_config['alpha_loss']
    gamma_loss = hp_config['gamma_loss']

    # ----------------------------------- Cross Validation ----------------------------------- #
    n_hidden_filters_config = hp_config['n_hidden_filters']
    n_hidden_filters = hp.Int('n_hidden_filters',
                            min_value = n_hidden_filters_config['min'],
                            max_value= n_hidden_filters_config['max'],
                            step=n_hidden_filters_config['step'])
    
    n_hidden_convs_config = hp_config['n_hidden_convs']
    n_hidden_convs = hp.Int('n_hidden_convs',
                            min_value = n_hidden_convs_config['min'],
                            max_value= n_hidden_convs_config['max'],
                            step=n_hidden_convs_config['step'])

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
    
    model = Model(inputs=input_final, outputs=predicted_init, name='Classic_Model')

    # Algoritmo de optimización Adam
    optimizer = Adam(learning_rate = learning_rate)
    model.compile(
        optimizer=optimizer,
        loss = mt.classic_customLoss(gamma=gamma_loss, alpha = alpha_loss),
        metrics= [mt.Accuracy(threshold=threshold, wl=False), 
                mt.Recall(threshold=threshold, wl=False),
                mt.Specificity(threshold=threshold, wl=False),
                mt.Precision(threshold=threshold, wl=False),
                mt.F1Score(threshold=threshold, wl=False), 
                mt.Absolute(threshold=threshold, wl=False)
                ])

    return model