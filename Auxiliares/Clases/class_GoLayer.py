import tensorflow as tf
import keras

@keras.saving.register_keras_serializable()
class ConwayLayer(tf.keras.layers.Layer):
    def __init__(self, delta=1, epsilon=50, order = 4, **kwargs):
        """
        Builder of the Differentiable Conway's Game of Life Layer. This layer is a continuous analogue of 
        Conway's discrete rules. Given a continuous initial state, it outputs a continuous final state.

        Arguments:
        
        delta (int): number of generations between the initial and final state.
        epsilon (float): steepness of the differentiable Conway's Game of Life rule function. Controls
        the strictness of aplication the target rule.
        order (float): exponent that controls the stricness of the binarisation.

        Outputs:

        Differentiable Conway's Game of Life layer (tensorflow layer).

        

        """
        super().__init__(**kwargs)
        self.delta = delta
        self.epsilon = epsilon
        self.order = order
        kernel = [[1, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]]
        self.kernel = tf.constant(kernel, dtype=tf.float32)
        self.kernel = tf.reshape(self.kernel, [3, 3, 1, 1]) 

    def periodic_padding(self, x):
        """
        Reproduce periodic boundary conditions.

        Arguments:

        x (array): state of size (1, H,W, 1)

        Outputs:

        x_padded (tf.tensor): state of size (1, H+2, W+2, 1)

        """
        # 1. Padding Vertical:
        # Concatenamos la última fila al principio y la primera fila al final
        upper_pad = x[:, -1:, :, :] # Toma la última fila
        lower_pad = x[:, :1, :, :]  # Toma la primera fila
        x_padded = tf.concat([upper_pad, x, lower_pad], axis=1)

        # 2. Padding Horizontal:
        # Concatenamos la última columna (del tensor ya estirado) a la izq y la primera a la der
        left_pad = x_padded[:, :, -1:, :]
        right_pad = x_padded[:, :, :1, :]
        x_padded = tf.concat([left_pad, x_padded, right_pad], axis=2)
        
        return x_padded

    def call(self, x):

        """
        Reproduce the continuous version of Conway's Game of Life during delta generations.

        Arguments:

        x (array): initial state of size (1, H,W, 1)

        Outputs:

        x (tf.tensor): final state of size (1, H+2, W+2, 1)

        """
        
        x = tf.cast(x, tf.float32)
        for _ in range(self.delta):
            # Aplico padding periódico manual
            x_padded = self.periodic_padding(x)
            
            # Convolución para contar vecinos vivos
            neighbors = tf.nn.conv2d(x_padded, self.kernel, strides=1, padding='VALID')
            
            # Lógica de actualización:
            survive = 2*(x**self.order)*(tf.sigmoid(-self.epsilon*((neighbors - 2)**2)) + tf.sigmoid(-self.epsilon*(neighbors - 3)**2) ) 
            born = 2*((1 - x)**self.order)* tf.sigmoid(-self.epsilon*(neighbors - 3) ** 2) 
            x = tf.clip_by_value(survive + born, 0, 1)
            
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({"delta": self.delta, "epsilon": self.epsilon, "order":self.order})
        return config