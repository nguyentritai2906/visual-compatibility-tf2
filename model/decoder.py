import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class Decoder(Layer):
    """
    MLP-based decoder model layer for edge-prediction.
    """

    def __init__(self, dropout_rate=0.0, use_bias=False, **kwargs):
        super(Decoder, self).__init__(**kwargs)

        self.dropout_rate = dropout_rate
        self.use_bias = use_bias

    def build(self, input_shape):
        self.w = tf.Variable(tf.random.uniform([input_shape[-1], 1], -1, 1))
        if self.use_bias:
            self.b = tf.Variable(tf.zeros([1]))
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, node_features, r_indices, c_indices):
        node_inputs = self.dropout(node_features)

        # r corresponds to the selected rows, and c to the selected columns
        row_inputs = tf.gather(node_inputs, tf.cast(r_indices, tf.int32))
        col_inputs = tf.gather(node_inputs, tf.cast(c_indices, tf.int32))

        diff = tf.abs(row_inputs - col_inputs)

        outputs = tf.matmul(diff, tf.cast(self.w, self.compute_dtype))

        if self.use_bias:
            outputs += tf.cast(self.b, self.compute_dtype)

        outputs = tf.squeeze(outputs)  # remove single dimension

        return outputs

    def get_config(self):
        config = {'dropout': self.dropout_rate, 'use_bias': self.use_bias}
        base_config = super(Decoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
