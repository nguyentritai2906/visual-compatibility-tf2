import tensorflow as tf
from tensorflow.keras.layers import Layer


class GCNLayer(Layer):
    """Graph convolution layer for multiple degree adjacencies"""

    def __init__(self,
                 output_dim,
                 num_support,
                 use_bias=False,
                 w_init='def',
                 **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        assert w_init in ['def', 'he']
        self.w_init = w_init
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.num_support = num_support

    def build(self, input_shape):
        if self.w_init == 'def':
            self.w = tf.Variable(
                tf.random.uniform(
                    [self.num_support, input_shape[-1], self.output_dim],
                    -0.05, 0.05))
        else:
            self.w = tf.Variable(
                tf.keras.initializers.he_normal(
                    [self.num_support, input_shape[-1], self.output_dim],
                    -0.05, 0.05))

        if self.use_bias:
            self.b = tf.Variable(tf.zeros([self.output_dim]))

    def call(self, x_n, supports):
        supports_n = tf.matmul(x_n, tf.cast(self.w, self.compute_dtype))

        z_n = tf.reduce_sum(supports_n, axis=0)

        if self.use_bias:
            z_n = tf.nn.bias_add(z_n, self.b)

        return z_n

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'num_support': self.num_support,
            'use_bias': self.use_bias,
            'w_init': self.w_init
        }
        base_config = super(GCNLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GCNBlock(Layer):

    def __init__(self,
                 output_dim,
                 num_support,
                 dropout_rate=0.0,
                 use_batch_norm=False,
                 **kwargs):
        super(GCNBlock, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.num_support = num_support
        self.use_batch_norm = use_batch_norm

    def build(self, input_shape):
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.gcn = GCNLayer(self.output_dim,
                            num_support=self.num_support,
                            use_bias=not self.use_batch_norm)
        self.act = tf.keras.layers.Activation('relu')
        if self.use_batch_norm:
            self.batch_norm = tf.keras.layers.BatchNormalization()

    def call(self, x, supports):
        x = self.dropout(x)
        x = self.gcn(x, supports)
        x = self.act(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        return x

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'num_support': self.num_support,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm
        }
        base_config = super(GCNBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Encoder(Layer):

    def __init__(self,
                 num_support,
                 num_hidden,
                 dropout_rate=0.0,
                 use_batch_norm=False,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.num_support = num_support
        self.num_hidden = num_hidden
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.blocks = []

    def build(self, input_shape):
        for i in range(len(self.num_hidden)):
            self.blocks.append(
                GCNBlock(output_dim=self.num_hidden[i],
                         num_support=self.num_support,
                         dropout_rate=self.dropout_rate,
                         use_batch_norm=self.use_batch_norm))

    def call(self, x, supports):
        for block in self.blocks:
            x = block(x, supports)
        return x

    def get_config(self):
        config = {
            'num_support': self.num_support,
            'num_hidden': self.num_hidden,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.use_batch_norm
        }
        base_config = super(Encoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
