import tensorflow as tf

class Discriminator:
    def __init__(self, n_input, n_noise):
        self.n_hidden = 256
        self.n_input = n_input
        self.n_noise = n_noise

    def net(self, inputs, reuse = False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            D_W1 = tf.get_variable('D_W1', [self.n_input, self.n_hidden], initializer = tf.random_normal_initializer(stddev=0.01))
            D_b1 = tf.get_variable('D_b1', [self.n_hidden], initializer = tf.constant_initializer(0))
            D_W2 = tf.get_variable('D_W2', [self.n_hidden, 1], initializer = tf.random_normal_initializer(stddev=0.01))
            D_b2 = tf.get_variable('D_b2', [1], initializer = tf.constant_initializer(0))
 
            hidden = tf.nn.relu(
                            tf.matmul(inputs, D_W1) + D_b1)
            output = tf.nn.sigmoid(
                            tf.matmul(hidden, D_W2) + D_b2)

        return output