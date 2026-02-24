import tensorflow as tf
import keras
import Auxiliares.Clases.class_GoLayer as GoLayer

### ------------------------------------------------------------------------------ ###
#########               LOSS GENÉRICA CON REGULARIZACIÓN BINARIA             ######### 
### ------------------------------------------------------------------------------ ###   
@tf.keras.utils.register_keras_serializable()
class customLoss(tf.keras.losses.Loss):
    def __init__(self, lambda_phys = 1.0, lambda_bin=1,gamma=2.0, alpha=0.75, **kwargs):
        """
        Computes a custom loss function for architectures with a differentiable
        Conway's Game of Life layer. It computes:

        - Focal Binary Cross-Entropy (FBCE) on initial states. (main_loss)
        - Binary Cross-Entropy (BCE) on final states. (physics_loss)
        - Binarisation Penalty term on initial states. (binary_loss)
        
        Arguments:
        lambda_phys (float): weight of the BCE loss.
        lambda_bin (float): weight of the Binarisation Penalty term.
        gamma (float): parameter of the FBCE loss. Controls the down-weighting rate of easy 
        examples, focusing the model on hard-to-classify cells.
        alpha (float): parameter of the FBCE loss. Controls the weight of the positive 
        (live) class to mitigate class imbalance.

        Outputs:
        Custom loss function: total loss of the predicted set of initial and final states.
        """

        super().__init__(**kwargs)
        if isinstance(lambda_phys, dict) and 'config' in lambda_phys:
            lambda_phys = lambda_phys['config']['value']
            self.lambda_phys = tf.cast(lambda_phys, dtype=tf.float32)
        else:
            self.lambda_phys = lambda_phys
        
        if isinstance(lambda_bin, dict) and 'config' in lambda_bin:
            lambda_bin = lambda_bin['config']['value']
            self.lambda_bin = tf.cast(lambda_bin, dtype=tf.float32)
        else:
            self.lambda_bin = lambda_bin
            
        self.gamma = gamma
        self.alpha = alpha
        
        # Loss:
        self.FBCE = keras.losses.BinaryFocalCrossentropy(gamma=self.gamma, alpha=self.alpha)
        self.BCE = keras.losses.BinaryCrossentropy()

    def call(self, Y_TRUE, Y_PRED):
        """
        Computes a custom loss function for architectures with a differentiable
        Conway's Game of Life layer.

        Arguments:
        Y_TRUE (array): ground truth.
        Y_PRED (array): predicted values

        Outputs:

        loss (float): total loss of the predicted set of initial and final states. 
        """

        # 1. Separar los tensores del output
        init_pred, fin_pred = tf.split(Y_PRED, num_or_size_splits=2, axis=-1)
        init_true, fin_true = tf.split(Y_TRUE, num_or_size_splits=2, axis=-1)
        # 2. Pérdidas de los iniciales
        main_loss = self.FBCE(init_true, init_pred)
        # 3. Pérdidas de los finales:
        physics_loss = self.BCE(fin_true, fin_pred)
        # 4. Binarización de la salida:
        binary_loss = 4*tf.reduce_mean(init_pred*(1 - init_pred))
        # 5. Juntamos
        return main_loss + self.lambda_phys*physics_loss + self.lambda_bin*binary_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "lambda_phys": self.lambda_phys,
            "lambda_bin": self.lambda_bin,
            "gamma": self.gamma,
            "alpha": self.alpha
        })
        return config


### ------------------------------------------------------------------------------ ###
        #########               LOSS CLÁSICA FBCE             ######### 
### ------------------------------------------------------------------------------ ### 

@tf.keras.utils.register_keras_serializable()
class classic_customLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.75, lambda_bin=0, **kwargs):
        """
        Computes a custom loss function for architectures without the differentiable
        Conway's Game of Life layer. It computes:

        - Focal Binary Cross-Entropy (FBCE) on initial states. (main_loss)
        - Binarisation Penalty term on initial states. (binary_loss)
        
        Arguments:
        
        lambda_bin (float): weight of the Binarisation Penalty term.
        gamma (float): parameter of the FBCE loss. Controls the down-weighting rate of easy 
        examples, focusing the model on hard-to-classify cells.
        alpha (float): parameter of the FBCE loss. Controls the weight of the positive 
        (live) class to mitigate class imbalance.

        Outputs:
        Custom loss function: total loss of the predicted set of initial and final states.
        """
      
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.lambda_bin = lambda_bin
        
        # Loss:
        self.FBCE = keras.losses.BinaryFocalCrossentropy(gamma=self.gamma, alpha=self.alpha)

    def call(self, Y_TRUE, Y_PRED):

        # 1. Pérdidas de los iniciales:
        main_loss = self.FBCE(Y_TRUE, Y_PRED)
        # 2. Calculo la penalización binaria:
        binary_loss = 4*tf.reduce_mean(Y_PRED*(1 - Y_PRED))
        return main_loss + self.lambda_bin* binary_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
            "lambda_bin": self.lambda_bin
        })
        return config


### ------------------------------------------------------------------------------ ###
       #########           ACCURACY SOBRE TABLEROS INICIALES           ######### 
### ------------------------------------------------------------------------------ ### 
@tf.keras.utils.register_keras_serializable()
class Accuracy(tf.keras.metrics.Metric):
    def __init__(self, name="accuracy", threshold=0.5, wl=False, **kwargs):
        super(Accuracy,self).__init__(name=name, **kwargs)

        """
        Computes the Accuracy metric on the initial states in models with or without
        differentiable Game of Life Layer.

        Arguments:
        threshold (float): umbralization threshold. If value > threshold => 1, else 0. 
        wl (Bool): set to True if the models incorporates the custom layer.  

        Returns:

        accuracy (float): accuracy metric on initial states.
        """

        self.threshold = threshold
        self.wl = wl
        self.acc_metric = tf.keras.metrics.BinaryAccuracy(threshold = self.threshold)

    def update_state(self, Y_TRUE, Y_PRED, sample_weight=None): # 

        if self.wl == True: # Tenemos layer => Hay finales e iniciales:
            real, _ = tf.split(Y_TRUE, num_or_size_splits=2, axis=-1)
            prediccion, _ = tf.split(Y_PRED, num_or_size_splits=2,axis=-1)
        elif self.wl == False:
            prediccion = Y_PRED
            real = Y_TRUE

        self.acc_metric.update_state(real, prediccion, sample_weight)
        
    def result(self):
        return self.acc_metric.result()

    def reset_state(self):
        self.acc_metric.reset_state()
        
    def get_config(self):
        # Permite guardar el modelo con el threshold
        config = super().get_config()
        config.update({"threshold": self.threshold, "wl": self.wl})
        return config


### ------------------------------------------------------------------------------ ###
       #########           ACCURACY SOBRE TABLEROS FINALES          ######### 
### ------------------------------------------------------------------------------ ### 
@tf.keras.utils.register_keras_serializable()
class AccuracyFin(tf.keras.metrics.Metric):
    """
        Computes the Accuracy metric on the final states in models with
        differentiable Game of Life Layer.

        Arguments:
        threshold (float): umbralization threshold. If value > threshold => 1, else 0. 
        
        Returns:

        acc_fin (tf.float): accuracy metric on final states.
        """
    def __init__(self, name="acc_fin", threshold=0.5, **kwargs):
        super(AccuracyFin,self).__init__(name=name, **kwargs)

        self.threshold = threshold
        self.acc_metric = tf.keras.metrics.BinaryAccuracy(threshold = self.threshold)

    def update_state(self, Y_TRUE, Y_PRED, sample_weight=None): 

        _, real = tf.split(Y_TRUE, num_or_size_splits=2, axis=-1)
        _, prediccion = tf.split(Y_PRED, num_or_size_splits=2,axis=-1)
        self.acc_metric.update_state(real, prediccion, sample_weight)
        
    def result(self):
        return self.acc_metric.result()

    def reset_state(self):
        self.acc_metric.reset_state()
        
    def get_config(self):
        # Permite guardar el modelo con el threshold
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config
    

### ------------------------------------------------------------------------------ ###
       #########           RECALL SOBRE TABLEROS INICIALES           ######### 
### ------------------------------------------------------------------------------ ### 

@tf.keras.utils.register_keras_serializable()
class Recall(tf.keras.metrics.Metric):
    def __init__(self, name="recall", threshold=0.5, wl=False, **kwargs):
        """
        Computes the Recall metric on the initial states in models with or without
        differentiable Game of Life Layer.

        Arguments:
        threshold (float): umbralization threshold. If value > threshold => 1, else 0.  
        wl (Bool): set to True if the models incorporates the custom layer.  

        Returns:

        recall (float): Recall metric on initial states.
        """
        super(Recall,self).__init__(name=name, **kwargs)

        self.threshold = threshold
        self.wl = wl
        self.rec_metric = tf.keras.metrics.Recall(thresholds = self.threshold)
        
    def update_state(self, Y_TRUE, Y_PRED, sample_weight=None): # 

        
        if self.wl == True: # Tenemos layer => Hay finales e iniciales:
            prediccion, _ = tf.split(Y_PRED, num_or_size_splits=2,axis=-1)
            real, _ = tf.split(Y_TRUE, num_or_size_splits=2, axis=-1)
        elif self.wl == False:
            prediccion = Y_PRED
            real = Y_TRUE
        self.rec_metric.update_state(real, prediccion, sample_weight)
        
    def result(self):
        return self.rec_metric.result()

    def reset_state(self):
        self.rec_metric.reset_state()
        
    def get_config(self):
        # Permite guardar el modelo con el threshold
        config = super().get_config()
        config.update({"threshold": self.threshold, "wl": self.wl})
        return config
    
### ------------------------------------------------------------------------------ ###
       #########           RECALL SOBRE TABLEROS FINALES          ######### 
### ------------------------------------------------------------------------------ ### 

@tf.keras.utils.register_keras_serializable()
class Recall_fin(tf.keras.metrics.Metric):
    def __init__(self, name="recall_fin", threshold=0.5, wl=False, **kwargs):
        """
        Computes the Recall metric on the final states in models with or without
        differentiable Game of Life Layer.

        Arguments:
        threshold (float): umbralization threshold. If value > threshold => 1, else 0.
        wl (Bool): set to True if the models incorporates the custom layer.  

        Returns:

        recall_fin (float): accuracy metric on final states.
        """
        super(Recall_fin,self).__init__(name=name, **kwargs)

        self.threshold = threshold
        self.wl = wl
        self.rec_metric = tf.keras.metrics.Recall(thresholds = self.threshold)
        
    def update_state(self, Y_TRUE, Y_PRED, sample_weight=None): # 

        _, real = tf.split(Y_TRUE, num_or_size_splits=2, axis=-1)
        if self.wl == True: # Tenemos layer => Hay finales e iniciales:
            _, prediccion = tf.split(Y_PRED, num_or_size_splits=2,axis=-1)
        elif self.wl == False:
            prediccion = Y_PRED

        self.rec_metric.update_state(real, prediccion, sample_weight)
        
    def result(self):
        return self.rec_metric.result()

    def reset_state(self):
        self.rec_metric.reset_state()
        
    def get_config(self):
        # Permite guardar el modelo con el threshold
        config = super().get_config()
        config.update({"threshold": self.threshold, "wl": self.wl})
        return config
    
### ------------------------------------------------------------------------------ ###
       #########           SPECIFICITY SOBRE TABLEROS INICIALES           ######### 
### ------------------------------------------------------------------------------ ### 

@tf.keras.utils.register_keras_serializable()
class Specificity(tf.keras.metrics.Metric):
    def __init__(self, name="specificity", threshold=0.5, wl=False, **kwargs):
        """
        Computes the Specificity metric on the initial states in models with or without
        differentiable Game of Life Layer.

        Arguments:
        threshold (float): umbralization threshold. If value > threshold => 1, else 0.
        wl (Bool): set to True if the models incorporates the custom layer.  

        Returns:

        specificity (float): specificity metric on initial states.
        """
        super(Specificity,self).__init__(name=name, **kwargs)

        self.threshold = threshold
        self.wl = wl
        self.tn = self.add_weight(name="tn", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        
    def update_state(self, Y_TRUE, Y_PRED, sample_weight=None): # 

        
        if self.wl == True: # Tenemos layer => Hay finales e iniciales:
            prediccion, _ = tf.split(Y_PRED, num_or_size_splits=2,axis=-1)
            real, _ = tf.split(Y_TRUE, num_or_size_splits=2, axis=-1)
        elif self.wl == False:
            prediccion = Y_PRED
            real = Y_TRUE

        cast_y_true = tf.cast(real, tf.bool)
        
        # 1. Invierto el cast_y_true porque busco los TN
        is_actual_negative = tf.math.logical_not(cast_y_true)

        # 2. Cálculo de TN (Verdaderos Negativos)
        # El modelo predice negativo si es MENOR al umbral
        # Hacemos uso de less_equal para no perder el umbral.
        pred_negative = tf.less_equal(prediccion, self.threshold)
        values_tn = tf.logical_and(is_actual_negative, pred_negative)
        
        # 3. Cálculo de FP (Falsos Positivos)
        # El modelo predice positivo si es MAYOR O IGUAL al umbral
        # Hacemos uso de greater para seguir la lógica de keras
        pred_positive = tf.greater(prediccion, self.threshold)
        values_fp = tf.logical_and(is_actual_negative, pred_positive)

        # 4. Acumulación
        # Convertimos a float para sumar
        tn_float = tf.cast(values_tn, self.dtype)
        fp_float = tf.cast(values_fp, self.dtype)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            tn_float = tf.multiply(tn_float, sample_weight)
            fp_float = tf.multiply(fp_float, sample_weight)

        self.tn.assign_add(tf.reduce_sum(tn_float))
        self.fp.assign_add(tf.reduce_sum(fp_float))
        
    def result(self):
        return tf.math.divide_no_nan(self.tn, self.tn + self.fp)

    def reset_state(self):
        self.tn.assign(0.0)
        self.fp.assign(0.0)
        
    def get_config(self):
        # Permite guardar el modelo con el threshold
        config = super().get_config()
        config.update({"threshold": self.threshold, "wl": self.wl})
        return config
    
### ------------------------------------------------------------------------------ ###
       #########           SPECIFICITY SOBRE TABLEROS finales           ######### 
### ------------------------------------------------------------------------------ ###     
@tf.keras.utils.register_keras_serializable()
class Specificity_fin(tf.keras.metrics.Metric):
    def __init__(self, name="specificity_fin", threshold=0.5, wl=False, **kwargs):
        """
        Computes the Specificity metric on the final states in models with or without
        differentiable Game of Life Layer.

        Arguments:
        threshold (float): umbralization threshold. If value > threshold => 1, else 0. 
        wl (Bool): set to True if the models incorporates the custom layer.  

        Returns:

        specificity_fin (float): specificity metric on initial states.
        """
        super(Specificity_fin,self).__init__(name=name, **kwargs)

        self.threshold = threshold
        self.wl = wl
        self.tn = self.add_weight(name="tn", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        
    def update_state(self, Y_TRUE, Y_PRED, sample_weight=None): # 

        _, real = tf.split(Y_TRUE, num_or_size_splits=2, axis=-1)
        if self.wl == True: # Tenemos layer => Hay finales e iniciales:
            _, prediccion = tf.split(Y_PRED, num_or_size_splits=2,axis=-1)
        elif self.wl == False:
            prediccion = Y_PRED

        cast_y_true = tf.cast(real, tf.bool)
        
        # 1. Invierto el cast_y_true porque busco los TN
        is_actual_negative = tf.math.logical_not(cast_y_true)

        # 2. Cálculo de TN (Verdaderos Negativos)
        # El modelo predice negativo si es MENOR al umbral
        # Hacemos uso de less_equal para no perder el umbral.
        pred_negative = tf.less_equal(prediccion, self.threshold)
        values_tn = tf.logical_and(is_actual_negative, pred_negative)
        
        # 3. Cálculo de FP (Falsos Positivos)
        # El modelo predice positivo si es MAYOR O IGUAL al umbral
        # Hacemos uso de greater para seguir la lógica de keras
        pred_positive = tf.greater(prediccion, self.threshold)
        values_fp = tf.logical_and(is_actual_negative, pred_positive)

        # 4. Acumulación
        # Convertimos a float para sumar
        tn_float = tf.cast(values_tn, self.dtype)
        fp_float = tf.cast(values_fp, self.dtype)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            tn_float = tf.multiply(tn_float, sample_weight)
            fp_float = tf.multiply(fp_float, sample_weight)

        self.tn.assign_add(tf.reduce_sum(tn_float))
        self.fp.assign_add(tf.reduce_sum(fp_float))
        
    def result(self):
        return tf.math.divide_no_nan(self.tn, self.tn + self.fp)

    def reset_state(self):
        self.tn.assign(0.0)
        self.fp.assign(0.0)
        
    def get_config(self):
        # Permite guardar el modelo con el threshold
        config = super().get_config()
        config.update({"threshold": self.threshold, "wl": self.wl})
        return config

### ------------------------------------------------------------------------------ ###
       #########           PRECISION SOBRE TABLEROS INICIALES           ######### 
### ------------------------------------------------------------------------------ ### 

@tf.keras.utils.register_keras_serializable()
class Precision(tf.keras.metrics.Metric):
    def __init__(self, name="precision", threshold=0.5, wl=False, **kwargs):
        """
        Computes the Precision metric on the initial states in models with or without
        differentiable Game of Life Layer.

        Arguments:
        threshold (float): umbralization threshold. If value > threshold => 1, else 0.
        wl (Bool): set to True if the models incorporates the custom layer.  

        Returns:

        precision (float): precision metric on initial states.
        """
        super(Precision,self).__init__(name=name, **kwargs)

        self.threshold = threshold
        self.wl = wl
        self.metric = tf.keras.metrics.Precision(thresholds = self.threshold)
        
    def update_state(self, Y_TRUE, Y_PRED, sample_weight=None): # 

        
        if self.wl == True: # Tenemos layer => Hay finales e iniciales:
            prediccion, _ = tf.split(Y_PRED, num_or_size_splits=2,axis=-1)
            real, _ = tf.split(Y_TRUE, num_or_size_splits=2, axis=-1)
        elif self.wl == False:
            prediccion = Y_PRED
            real = Y_TRUE

        self.metric.update_state(real, prediccion, sample_weight)
        
    def result(self):
        return self.metric.result()

    def reset_state(self):
        self.metric.reset_state()
        
    def get_config(self):
        # Permite guardar el modelo con el threshold
        config = super().get_config()
        config.update({"threshold": self.threshold, "wl": self.wl})
        return config
    
### ------------------------------------------------------------------------------ ###
       #########           PRECISION SOBRE TABLEROS Finales           ######### 
### ------------------------------------------------------------------------------ ### 

@tf.keras.utils.register_keras_serializable()
class Precision_fin(tf.keras.metrics.Metric):
    def __init__(self, name="precision_fin", threshold=0.5, wl=False, **kwargs):
        """
        Computes the Precision metric on the final states in models with or without
        differentiable Game of Life Layer.

        Arguments:
        threshold (float): umbralization threshold. If value > threshold => 1, else 0.
        wl (Bool): set to True if the models incorporates the custom layer.  

        Returns:

        precision_fin (float): precision metric on final states.
        """
        super(Precision_fin,self).__init__(name=name, **kwargs)

        self.threshold = threshold
        self.wl = wl
        self.metric = tf.keras.metrics.Precision(thresholds = self.threshold)
        
    def update_state(self, Y_TRUE, Y_PRED, sample_weight=None): # 

        _, real = tf.split(Y_TRUE, num_or_size_splits=2, axis=-1)
        if self.wl == True: # Tenemos layer => Hay finales e iniciales:
            _, prediccion = tf.split(Y_PRED, num_or_size_splits=2,axis=-1)
        elif self.wl == False:
            prediccion = Y_PRED

        self.metric.update_state(real, prediccion, sample_weight)
        
    def result(self):
        return self.metric.result()

    def reset_state(self):
        self.metric.reset_state()
        
    def get_config(self):
        # Permite guardar el modelo con el threshold
        config = super().get_config()
        config.update({"threshold": self.threshold, "wl": self.wl})
        return config
### ------------------------------------------------------------------------------ ###
       #########           F1-Score SOBRE TABLEROS INICIALES           ######### 
### ------------------------------------------------------------------------------ ### 

@tf.keras.utils.register_keras_serializable()
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="F1score", threshold=0.5, wl=False, **kwargs):
        super(F1Score,self).__init__(name=name, **kwargs)
        """
        Computes the F1-Score metric on the initial states in models with or without
        differentiable Game of Life Layer.

        Arguments:
        threshold (float): umbralization threshold. If value > threshold => 1, else 0.
        wl (Bool): set to True if the models incorporates the custom layer.  

        Returns:

        F1score (float): F1-Score metric on initial states.
        """
        self.threshold = threshold
        self.wl = wl
        # Initialize variables to store TP, FP, FN
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')
        
    def update_state(self, Y_TRUE, Y_PRED, sample_weight=None): # 

        
        if self.wl == True: # Tenemos layer => Hay finales e iniciales:
            prediccion, _ = tf.split(Y_PRED, num_or_size_splits=2,axis=-1)
            real, _ = tf.split(Y_TRUE, num_or_size_splits=2, axis=-1)
        elif self.wl == False:
            prediccion = Y_PRED
            real = Y_TRUE

        # 2. Cast to bool/float based on threshold
        real = tf.cast(real, tf.bool)
        prediccion = tf.greater(prediccion, self.threshold)

        # 3. Calculate intermediate booleans
        # True Positives: Real is True AND Pred is True
        values_tp = tf.logical_and(real, prediccion)
        # False Positives: Real is False AND Pred is True
        values_fp = tf.logical_and(tf.logical_not(real), prediccion)
        # False Negatives: Real is True AND Pred is False
        values_fn = tf.logical_and(real, tf.logical_not(prediccion))

        # 4. Cast to float for summation
        tp_float = tf.cast(values_tp, self.dtype)
        fp_float = tf.cast(values_fp, self.dtype)
        fn_float = tf.cast(values_fn, self.dtype)

        # 5. Apply sample weights if they exist
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            # Broadcast weights if necessary
            tp_float = tf.multiply(tp_float, sample_weight)
            fp_float = tf.multiply(fp_float, sample_weight)
            fn_float = tf.multiply(fn_float, sample_weight)

        # 6. Update state
        self.tp.assign_add(tf.reduce_sum(tp_float))
        self.fp.assign_add(tf.reduce_sum(fp_float))
        self.fn.assign_add(tf.reduce_sum(fn_float))

        
    def result(self):
        # F1 Formula: 2*TP / (2*TP + FP + FN)
        # Add epsilon to denominator to avoid division by zero
        numerator = 2.0 * self.tp
        denominator = (2.0 * self.tp) + self.fp + self.fn 
        
        return tf.math.divide_no_nan(numerator, denominator)
    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)
        
    def get_config(self):
        # Permite guardar el modelo con el threshold
        config = super().get_config()
        config.update({"threshold": self.threshold, "wl": self.wl})
        return config
    
### ------------------------------------------------------------------------------ ###
       #########           ABSOLUTA SOBRE TABLEROS INICIALES           ######### 
### ------------------------------------------------------------------------------ ### 
@tf.keras.utils.register_keras_serializable()
class Absolute(tf.keras.metrics.Metric):
    def __init__(self, name="absolute", threshold=0.5, wl=False, **kwargs):
        """
        Computes the Absolute metric on the initial states in models with or without
        differentiable Game of Life Layer. This metric computes the number of predicted initial 
        states that are complete matches of their ground truth.

        Arguments:
        threshold (float): umbralization threshold. If value > threshold => 1, else 0.
        wl (Bool): set to True if the models incorporates the custom layer.  

        Returns:

        absolute (float): absolute metric on initial states.
        """
        super(Absolute,self).__init__(name=name, **kwargs)

        self.threshold = threshold
        self.wl = wl
        self.total_matches = self.add_weight(name='matches', initializer='zeros') # Contador de match
        self.count = self.add_weight(name='count', initializer='zeros') # Contador de tableros
    
        
    def update_state(self, Y_TRUE, Y_PRED, sample_weight=None): # 

        
        if self.wl == True: # Tenemos layer => Hay finales e iniciales:
            prediccion, _ = tf.split(Y_PRED, num_or_size_splits=2,axis=-1)
            real, _ = tf.split(Y_TRUE, num_or_size_splits=2, axis=-1)
        elif self.wl == False:
            prediccion = Y_PRED
            real = Y_TRUE

        cast_y_true = tf.cast(real, tf.bool) # Convierto a booleans el real.
        cast_y_pred = tf.cast(tf.greater(prediccion, self.threshold), tf.bool) # El modelo predice VIVO si y_pred es MAYOR al threshold

        are_equal = tf.equal(cast_y_true, cast_y_pred) # Comparación elemento a elemento
        are_equal_flat = tf.reshape(are_equal, (tf.shape(cast_y_true)[0], -1)) # Aplanamos a (batch, height*width)
        is_perfect_sample = tf.math.reduce_all(are_equal_flat, axis=1) # Comparación final fila a fila

        match_values = tf.cast(is_perfect_sample, self.dtype) # 1.0 si acierto, 0.0 si fallo
        count_values = tf.ones_like(match_values)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
                
            # Multiplicamos el acierto por el peso (ej: Acierto(1.0) * Peso(2.0) = 2.0 puntos)
            match_values = tf.multiply(match_values, sample_weight)
            
            # El contador total también debe reflejar el peso (ej: Hemos visto "2.0 unidades")
            count_values = tf.multiply(count_values, sample_weight)

        # Actualizamos
        self.total_matches.assign_add(tf.reduce_sum(match_values)) 
        self.count.assign_add(tf.reduce_sum(count_values))
        
    def result(self):
        return tf.math.divide_no_nan(self.total_matches, self.count)

    def reset_state(self):
        self.total_matches.assign(0.0)
        self.count.assign(0.0)
        
    def get_config(self):
        # Permite guardar el modelo con el threshold
        config = super().get_config()
        config.update({"threshold": self.threshold, "wl": self.wl})
        return config
    

### ------------------------------------------------------------------------------ ###
        #########               COMPONENTE FBCE             ######### 
### ------------------------------------------------------------------------------ ###   
@tf.keras.utils.register_keras_serializable()
class ComponenteFBCE(tf.keras.metrics.Mean):  
    def __init__(self, name='componenteFBCE', gamma=2.0, alpha=0.75, **kwargs):
        """
        Computes the FBCE term in the custom Loss Functions in models with 
        differentiable Game of Life Layer. 

        Arguments:
        gamma (float): parameter of the FBCE loss. Controls the down-weighting rate of easy 
        examples, focusing the model on hard-to-classify cells.

        alpha (float): parameter of the FBCE loss. Controls the weight of the positive 
        (live) class to mitigate class imbalance.

        Returns:

        componenteFBCE (float): FBCE term in the custom Loss Function.
        """
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        
        self.fbce_fn = keras.losses.BinaryFocalCrossentropy(
            gamma=self.gamma, 
            alpha=self.alpha,
            reduction='none'
        )

    
    def update_state(self, y_true, y_pred, sample_weight=None):
        

        init_pred, _ = tf.split(y_pred, num_or_size_splits=2, axis=-1)
        init_true, _ = tf.split(y_true, num_or_size_splits=2, axis=-1)
    
        matches_loss = self.fbce_fn(init_true, init_pred)

        return super().update_state(matches_loss, sample_weight=sample_weight)

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha
        })
        return config
    
### ------------------------------------------------------------------------------ ###
        #########               COMPONENTE BCE             ######### 
### ------------------------------------------------------------------------------ ###   
@tf.keras.utils.register_keras_serializable()
class ComponenteBCE(tf.keras.metrics.Mean):  
    def __init__(self, name='componenteBCE', lambda_phys=1, **kwargs):
        """
        Computes the BCE term in the custom Loss Functions in models with 
        differentiable Game of Life Layer. 

        Arguments:
        lambda_phys (float): weight of the BCE loss.

        Returns:

        componenteBCE (float): BCE term in the custom Loss Function.
        """
        super().__init__(name=name, **kwargs)

        if isinstance(lambda_phys, dict) and 'config' in lambda_phys:
            lambda_phys = lambda_phys['config']['value']
            self.lambda_phys = tf.cast(lambda_phys, dtype=tf.float32)
        else:
            self.lambda_phys = lambda_phys
        
        self.bce_fn = keras.losses.BinaryCrossentropy(reduction='none')

    def update_state(self, y_true, y_pred, sample_weight=None):
        
        _, fin_pred = tf.split(y_pred, num_or_size_splits=2, axis=-1)
        _, fin_true = tf.split(y_true, num_or_size_splits=2, axis=-1)
    
        matches_loss = self.lambda_phys * self.bce_fn(fin_true, fin_pred)

        return super().update_state(matches_loss, sample_weight=sample_weight)

    def get_config(self):
        config = super().get_config()
        config.update({
            "lambda_phys": self.lambda_phys
        })
        return config
    

### ------------------------------------------------------------------------------ ###
        #########               COMPONENTE BCE             ######### 
### ------------------------------------------------------------------------------ ###   
@tf.keras.utils.register_keras_serializable()
class ComponenteBinarization(tf.keras.metrics.Mean):  
    def __init__(self, name='componenteBinarization', lambda_bin=1, **kwargs):
        """
        Computes the Binarisation Penalty term in the custom Loss Functions in models with 
        differentiable Game of Life Layer. 

        Arguments:
        lambda_bin (float): weight of the Binarisation Penalty term.

        Returns:

        componenteBCE (float): BCE term in the custom Loss Function.
        """
        super().__init__(name=name, **kwargs)
        
        
        if isinstance(lambda_bin, dict) and 'config' in lambda_bin:
            lambda_bin = lambda_bin['config']['value']
            self.lambda_bin = tf.cast(lambda_bin, dtype=tf.float32)
        else:
            self.lambda_bin = lambda_bin

    def update_state(self, y_true, y_pred, sample_weight=None):
        
        init_pred, _ = tf.split(y_pred, num_or_size_splits=2, axis=-1)
        init_true, _ = tf.split(y_true, num_or_size_splits=2, axis=-1)
    
        matches_loss = self.lambda_bin * init_pred*(1 - init_pred)

        return super().update_state(matches_loss, sample_weight=sample_weight)

    def get_config(self):
        config = super().get_config()
        config.update({
            "lambda_bin": self.lambda_bin
        })
        return config
    