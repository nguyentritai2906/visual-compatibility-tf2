import argparse
import time

import numpy as np
import tensorflow as tf
from dataloaders.SeqPolyvore import SeqPolyvore
from model.CompatibilityGAE import CompatibilityGAE
from model.metrics import sigmoid_accuracy, sigmoid_cross_entropy
from tensorflow.keras import mixed_precision
from utils import get_degree_supports, normalize_nonsym_adj, support_dropout

# mixed_precision.set_global_policy('mixed_float16')

# Set random seed
seed = int(time.time())  # 12342
np.random.seed(seed)
tf.random.set_seed(seed)


def parse_arguments():
    # Settings
    ap = argparse.ArgumentParser()
    ap.add_argument("-d",
                    "--dataset",
                    type=str,
                    default="polyvore",
                    choices=['fashiongen', 'polyvore', 'amazon'],
                    help="Dataset string.")

    ap.add_argument("-lr",
                    "--learning_rate",
                    type=float,
                    default=0.001,
                    help="Learning rate")

    ap.add_argument("-wd",
                    "--weight_decay",
                    type=float,
                    default=0.,
                    help="Learning rate")

    ap.add_argument("-e",
                    "--epochs",
                    type=int,
                    default=4000,
                    help="Number training epochs")

    ap.add_argument("-hi",
                    "--hidden",
                    type=int,
                    nargs='+',
                    default=[350, 350, 350],
                    help="Number hidden units in the GCN layers.")

    ap.add_argument("-do",
                    "--dropout",
                    type=float,
                    default=0.5,
                    help="Dropout fraction")

    ap.add_argument("-deg",
                    "--degree",
                    type=int,
                    default=1,
                    help="Degree of the convolution (Number of supports)")
    ap.add_argument("-m",
                    "--mode",
                    type=str,
                    default="fit",
                    choices=['fit', 'eager'],
                    help="Mode of execution.")

    ap.add_argument(
        "-sup_do",
        "--support_dropout",
        type=float,
        default=0.15,
        help=
        "Use dropout on the support matrices, dropping all the connections from some nodes"
    )

    fp = ap.add_mutually_exclusive_group(required=False)
    fp.add_argument('-bn',
                    '--batch_norm',
                    dest='batch_norm',
                    help="Option to turn on batchnorm in GCN layers",
                    action='store_true')
    ap.set_defaults(batch_norm=True)

    ap.add_argument("-amzd",
                    "--amz_data",
                    type=str,
                    default="Men_bought_together",
                    choices=[
                        'Men_also_bought', 'Women_also_bought',
                        'Women_bought_together', 'Men_bought_together'
                    ],
                    help="Dataset string.")

    args = vars(ap.parse_args())
    return args


def main(args):

    print('Settings:')
    print(args, '\n')

    # Define parameters
    DATASET = args['dataset']
    NB_EPOCH = args['epochs']
    DO = args['dropout']
    HIDDEN = args['hidden']
    LR = args['learning_rate']
    DEGREE = args['degree']
    BATCH_NORM = args['batch_norm']
    SUP_DO = args['support_dropout']
    ADJ_SELF_CONNECTIONS = True
    MODE = args['mode']

    # Define dataset
    train_ds = SeqPolyvore('./data/polyvore/dataset/', 'train', SUP_DO, DEGREE,
                           ADJ_SELF_CONNECTIONS)
    mean, std = train_ds.get_moments()
    val_ds = SeqPolyvore('./data/polyvore/dataset/', 'valid', SUP_DO, DEGREE,
                         ADJ_SELF_CONNECTIONS, mean, std)

    model = CompatibilityGAE(num_support=train_ds.num_support,
                             num_hidden=HIDDEN,
                             dropout_rate=DO,
                             use_batch_norm=BATCH_NORM)
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR,
                                         beta_1=0.9,
                                         beta_2=0.999,
                                         epsilon=1.e-8)

    if MODE == 'eager':
        for epoch in range(NB_EPOCH):
            train_inputs, labels = train_ds[epoch]
            with tf.GradientTape() as tape:
                logist = model(train_inputs)
                loss = sigmoid_cross_entropy(logist, labels)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_acc = sigmoid_accuracy(logist, labels)

            val_inputs, val_labels = val_ds[epoch]
            val_logist = model(val_inputs, training=False)
            val_loss = sigmoid_cross_entropy(val_logist, val_labels)
            val_acc = sigmoid_accuracy(val_logist, val_labels)

            print(
                "Epoch: {}/{}, Loss: {:.04f}, Accuracy {:.04f}, Val Loss: {:.04f}, Val Accuracy: {:.04f}"
                .format(epoch, NB_EPOCH, loss.numpy(), train_acc.numpy(),
                        val_loss.numpy(), val_acc.numpy()))
    else:
        model.compile(optimizer=optimizer,
                      loss=sigmoid_cross_entropy,
                      metrics=[sigmoid_accuracy])
        model.fit(train_ds, epochs=NB_EPOCH, validation_data=val_ds)


if __name__ == "__main__":
    main(parse_arguments())
