#!/usr/bin/env python3
# TEAM MEMBERS:
# Antonio Krizmanic - 2b193238-8e3c-11ec-986f-f39926f24a9c
# Janek Putz - e31a3cae-8e6c-11ec-986f-f39926f24a9c
import argparse
import datetime
import os
import re
from typing import Dict
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--cnn", default="C-8-3-5-same,C-8-3-2-valid,F,H-50", type=str, help="CNN architecture.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


# The neural network model
class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        # : Create the model. The template uses functional API, but
        # feel free to use subclassing if you want.
        inputs = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])

        def extract_layer_strings(cnn_str):
            # Handle residual blocks special (assumes that no residual block is nested in another residual block)
            findings = re.findall('R-\[.*]', cnn_str)
            residual_blocks = []
            for r in findings:
                # will leave two ,, behind each other
                cnn_str = cnn_str.replace(r, '')
                r = r.replace("R-[", '')
                r = r.replace("]", '')
                residual_blocks.append(["R", [l.split("-") for l in r.split(",")]])
            residual_blocks_used = 0

            # split other layers and create list together with residual blocks
            layers = []
            for layer in cnn_str.split(','):
                if layer != '':
                    layers.append(layer.split("-"))
                else:
                    layers.append(residual_blocks[residual_blocks_used])
                    residual_blocks_used += 1
            return layers

        layer_strings = extract_layer_strings(args.cnn)
        # print(layer_strings)

        # : Add CNN layers specified by `args.cnn`, which contains
        # comma-separated list of the following layers:
        # - `C-filters-kernel_size-stride-padding`: Add a convolutional layer with ReLU
        #   activation and specified number of filters, kernel size, stride and padding.
        # - `CB-filters-kernel_size-stride-padding`: Same as `C`, but use batch normalization.
        #   In detail, start with a convolutional layer without bias and activation,
        #   then add batch normalization layer, and finally ReLU activation.
        # - `M-pool_size-stride`: Add max pooling with specified size and stride, using
        #   the default "valid" padding.
        # - `R-[layers]`: Add a residual connection. The `layers` contain a specification
        #   of at least one convolutional layer (but not a recursive residual connection `R`).
        #   The input to the `R` layer should be processed sequentially by `layers`, and the
        #   produced output (after the ReLU nonlinearty of the last layer) should be added
        #   to the input (of this `R` layer).
        # - `F`: Flatten inputs. Must appear exactly once in the architecture.
        # - `H-hidden_layer_size`: Add a dense layer with ReLU activation and specified size.
        # - `D-dropout_rate`: Apply dropout with the given dropout rate.
        # You can assume the resulting network is valid; it is fine to crash if it is not.
        #
        # Produce the results in variable `hidden`.

        # in case of CB: in conv2d with_bias=False
        # for R block - add input of this block to output of this block
        # hidden = tf.keras.layers.Flatten()(inputs)

        hidden = inputs

        def create_layers(layers, start_layer):
            h = start_layer
            for l in layers:
                if l[0] == 'C':
                    h = tf.keras.layers.Conv2D(filters=int(l[1]), kernel_size=int(l[2]), strides=int(l[3]), padding=l[4])(h)
                elif l[0] == 'CB':
                    h = tf.keras.layers.Conv2D(filters=int(l[1]), kernel_size=int(l[2]), strides=int(l[3]), padding=l[4], use_bias=False)(h)
                    h = tf.keras.layers.BatchNormalization()(h)
                    h = tf.keras.layers.ReLU()(h)
                elif l[0] == 'M':
                    h = tf.keras.layers.MaxPooling2D(pool_size=int(l[1]), strides=int(l[2]), padding="valid")(h)
                elif l[0] == 'R':
                    hidden_before = h
                    # call function recursively with layers listed in residual block
                    h = create_layers(l[1], h)
                    h = tf.keras.layers.Add()([hidden_before, h])
                    # h = tf.keras.layers.ReLU()(h)  # TODO: necessary?
                elif l[0] == 'F':
                    h = tf.keras.layers.Flatten()(h)
                elif l[0] == 'H':
                    h = tf.keras.layers.Dense(units=int(l[1]), activation=tf.nn.relu)(h)
                elif l[0] == 'D':
                    h = tf.keras.layers.Dropout(rate=float(l[1]))(h)
            return h

        hidden = create_layers(layer_strings, hidden)
        # Add the final output layer
        outputs = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)(hidden)

        super().__init__(inputs=inputs, outputs=outputs)
        self.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        # self.summary()
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1)


def main(args: argparse.Namespace) -> Dict[str, float]:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))[:99]

    # Load the data
    mnist = MNIST()

    # Create the model and train it
    model = Model(args)

    logs = model.fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        callbacks=[model.tb_callback],
    )

    # Return development metrics for ReCodEx to validate
    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
