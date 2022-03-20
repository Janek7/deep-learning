#!/usr/bin/env python3
# TEAM MEMBERS:
# Antonio Krizmanic - 2b193238-8e3c-11ec-986f-f39926f24a9c
# Janek Putz - e31a3cae-8e6c-11ec-986f-f39926f24a9c

import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from cifar10 import CIFAR10

# : Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=256, type=int, help="Batch size.")
parser.add_argument("--epochs", default=15, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

parser.add_argument("--conv_layers", default=3, type=int, help="Number of conv layers.")
parser.add_argument("--filters", default=64, type=int, help="Number of filters/channels to use in one conv layer.")
parser.add_argument("--kernel_size", default=3, type=int, help="Kernel size.")
parser.add_argument("--stride", default=1, type=int, help="Stride in conv layers.")
parser.add_argument("--padding", default="valid", type=str, help="Padding in conv layers.")
parser.add_argument("--activation_function", default="selu", type=str, help="Activation function")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay / L2 regularization")
parser.add_argument("--batch_norm", default=False, type=float, help="Flag for batch normalization")
parser.add_argument("--pooling", default="max", type=str, help="Type of pooling after each conv layer.")
parser.add_argument("--pooling_size", default=2, type=int, help="Size of downscaling in pooling layer")
parser.add_argument("--pooling_strides", default=2, type=int, help="Stride in pooling layer (usually == pooling_size")
parser.add_argument("--final_avg_pooling", default=False, type=bool, help="Flag to use average pooling instead of dense layers.")
parser.add_argument("--dense_layers", default=2, type=int, help="Number of dense layers.")
parser.add_argument("--dense_size", default=128, type=int, help="Size of dense layers.")
parser.add_argument("--dropout", default=0, type=float, help="Dropout rate for dropout layers after dense layers.")


def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load data
    cifar = CIFAR10()

    # findings: (IMPORTANT: can change with other changes in architecture)
    # - batch size 256 works the best
    # - dropout in dense layers does not improve
    # - doubling channels step by step after max pooling layers does not improve
    # - 3 conv layers are not deep enough to use global average pooling instead of dense layers

    # : Create the model and train it
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C]))

    # conv layers
    max_pool_mul = 1
    regularizer = tf.keras.regularizers.L2(args.weight_decay) if args.weight_decay else None
    activation = args.activation_function if not args.batch_norm else None

    for i in range(args.conv_layers):
        model.add(tf.keras.layers.Conv2D(filters=args.filters*max_pool_mul, kernel_size=args.kernel_size, strides=args.stride,
                                         padding=args.padding, kernel_regularizer=regularizer, activation=activation))
        if args.batch_norm:
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.ReLU())
        # add max pooling only if pooling is active. If average pooling is applied after the conv blocks instead of
        # dense layers dont add last max pooling layer
        if args.pooling and not (args.final_avg_pooling and i != args.conv_layers - 1):
            pool_class = tf.keras.layers.MaxPooling2D if args.pooling == 'max' else tf.keras.layers.AveragePooling2D
            model.add(pool_class(pool_size=args.pooling_size, strides=args.pooling_strides))
            max_pool_mul *= 2

    # average pooling OR dense layers
    if not args.final_avg_pooling:
        model.add(tf.keras.layers.Flatten())
        for i in range(args.dense_layers):
            model.add(tf.keras.layers.Dense(args.dense_size, activation=activation))
            if args.batch_norm:
                model.add(tf.keras.layers.BatchNormalization())
                model.add(tf.keras.layers.ReLU())
            model.add(tf.keras.layers.Dropout(args.dropout))
    else:
        model.add(tf.keras.layers.AveragePooling2D(pool_size=args.pooling_size, strides=args.pooling_strides))
        model.add(tf.keras.layers.Flatten())

    # final output layer
    model.add(tf.keras.layers.Dense(CIFAR10.LABELS, activation=tf.nn.softmax))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )
    model.summary()

    # model = tf.keras.models.load_model("C:\\Users\\janek\\Development\\Git\\Prag\\deep-learning-lecture\\04_cnns\\logs\\cifar_competition.py-2022-03-14_160914-bs=256,cl=3,dl=2,ds=128,d=0,e=10,f=64,ks=3,p=valid,p=max,ps=2,ps=2,s=42,s=1,t=1\\cifar_comp_model.h5")

    history = model.fit(
        cifar.train.data["images"], cifar.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(cifar.dev.data["images"], cifar.dev.data["labels"]),
        callbacks=[tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)],
    )

    model.save(os.path.join(args.logdir, 'cifar_comp_model.h5'), include_optimizer=True)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as predictions_file:
        for probs in model.predict(cifar.test.data["images"], batch_size=args.batch_size):
            print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
