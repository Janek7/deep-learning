#!/usr/bin/env python3
# TEAM MEMBERS:
# Antonio Krizmanic - 2b193238-8e3c-11ec-986f-f39926f24a9c
# Jan Kilic - 079f5b80-9807-11ec-986f-f39926f24a9c
# Janek Putz - e31a3cae-8e6c-11ec-986f-f39926f24a9c

import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--dropout", default=0, type=float, help="Dropout regularization.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layers", default=[400], nargs="*", type=int, help="Hidden layer sizes.")
parser.add_argument("--l2", default=0.001, type=float, help="L2 regularization.")
parser.add_argument("--label_smoothing", default=0, type=float, help="Label smoothing.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace) -> float:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load data
    mnist = MNIST(size={"train": 5000})

    # : Create the model and incorporate the L2 regularization and dropout:
    # - L2 regularization:
    #   If `args.l2` is nonzero, create a `tf.keras.regularizers.L2` regularizer
    #   and use it for all kernels (but not biases) of all Dense layers.
    # - Dropout:
    #   Add a `tf.keras.layers.Dropout` with `args.dropout` rate after the Flatten
    #   layer and after each Dense hidden layer (but not after the output Dense layer).

    if args.l2:
        regularizer = tf.keras.regularizers.L2(args.l2)
    else:
        regularizer = None

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=[MNIST.H, MNIST.W, MNIST.C]))
    model.add(tf.keras.layers.Dropout(rate=args.dropout))
    for hidden_layer in args.hidden_layers:
        model.add(tf.keras.layers.Dense(hidden_layer, activation=tf.nn.relu, kernel_regularizer=regularizer))
        model.add(tf.keras.layers.Dropout(rate=args.dropout))
    model.add(tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax))

    # : Implement label smoothing.
    # Apply the given smoothing. You will need to change the
    # `SparseCategorical{Crossentropy,Accuracy}` to `Categorical{Crossentropy,Accuracy}`
    # because `label_smoothing` is supported only by `CategoricalCrossentropy`.
    # That means you also need to modify the labels of all three datasets
    # (i.e., `mnist.{train,dev,test}.data["labels"]`) from indices of the gold class
    # to a full categorical distribution (you can use either NumPy or there is
    # a helper method also in the `tf.keras.utils`).
    mnist.train.data["labels"] = tf.keras.utils.to_categorical(mnist.train.data["labels"])
    mnist.dev.data["labels"] = tf.keras.utils.to_categorical(mnist.dev.data["labels"])
    mnist.test.data["labels"] = tf.keras.utils.to_categorical(mnist.test.data["labels"])

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
        metrics=[tf.metrics.CategoricalAccuracy(name="accuracy")],
    )

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

    def evaluate_test(epoch, logs):
        if epoch + 1 == args.epochs:
            test_logs = model.evaluate(
                mnist.test.data["images"], mnist.test.data["labels"], batch_size=args.batch_size, return_dict=True, verbose=0,
            )
            logs.update({"val_test_" + name: value for name, value in test_logs.items()})

    logs = model.fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=evaluate_test)],
    )

    # Return test accuracy for ReCodEx to validate
    return logs.history["val_test_accuracy"][-1]

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
