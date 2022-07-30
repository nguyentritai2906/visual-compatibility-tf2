import tensorflow as tf
from tensorflow.keras import Input, Model

from .decoder import Decoder
from .encoder import Encoder


class CompatibilityGAE(Model):

    def __init__(self,
                 num_support,
                 num_hidden,
                 dropout_rate,
                 use_batch_norm=False,
                 **kwargs):
        super(CompatibilityGAE, self).__init__(**kwargs)

        self.num_support = num_support
        self.num_hidden = num_hidden
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.encoder = Encoder(num_support=self.num_support,
                               num_hidden=self.num_hidden,
                               dropout_rate=self.dropout_rate,
                               use_batch_norm=self.use_batch_norm)
        self.decoder = Decoder(dropout_rate=0.0, use_bias=True)

    def call(self, inputs):
        x, supports, r_indices, c_indices = inputs
        x = self.encoder(x, supports)
        x = self.decoder(x, r_indices, c_indices)
        return x

    def summary(self):
        x = Input(shape=(84497, 2048))
        supports = Input(shape=(2, 84497, 84497))
        r_indices = Input(shape=(338488, ))
        c_indices = Input(shape=(338488, ))
        model = Model(inputs=[(x, supports, r_indices, c_indices)],
                      outputs=self.call((x, supports, r_indices, c_indices)))
        return model.summary()

    def get_config(self):
        config = super(CompatibilityGAE, self).get_config()
        config.update({
            'num_support': self.num_support,
            'num_hidden': self.num_hidden,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm
        })
        return config


if __name__ == "__main__":
    model = CompatibilityGAE(num_support=2,
                             num_hidden=[350, 350, 350],
                             dropout_rate=0.5)
    model.summary()
