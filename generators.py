import tensorflow as tf

class Generator:
    def __init__(self, n_input, n_noise):
        self.n_hidden = 256
        self.n_input = n_input
        self.n_noise = n_noise 

    def net(self, noise_z):
        with tf.variable_scope("generator"):
            G_W1 = tf.get_variable('G_W1', [self.n_noise, self.n_hidden], initializer = tf.random_normal_initializer(stddev=0.01))
            G_b1 = tf.get_variable('G_b1', [self.n_hidden], initializer = tf.constant_initializer(0))
            G_W2 = tf.get_variable('G_W2', [self.n_hidden, self.n_input], initializer = tf.random_normal_initializer(stddev=0.01))
            G_b2 = tf.get_variable('G_b2', [self.n_input], initializer = tf.constant_initializer(0)) 
            hidden = tf.nn.relu(
                    tf.matmul(noise_z, G_W1) + G_b1)
            output = tf.nn.sigmoid(
                    tf.matmul(hidden, G_W2) + G_b2)

        return output