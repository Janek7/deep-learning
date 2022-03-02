#!/usr/bin/env python3
# TEAM MEMBERS:
# Antonio Krizmanic - 2b193238-8e3c-11ec-986f-f39926f24a9c
# Jan Kilic - 079f5b80-9807-11ec-986f-f39926f24a9c
# Janek Putz - e31a3cae-8e6c-11ec-986f-f39926f24a9c

import argparse
import datetime
import os
from pickletools import optimize
import re

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--decay", default='linear', type=str, help="Learning decay rate type")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer", default=200, type=int, help="Size of the hidden layer.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default=0.0001, type=float, help="Final learning rate.")
parser.add_argument("--momentum", default=None, type=float, help="Momentum.")
parser.add_argument("--optimizer", default="Adam", type=str, help="Optimizer to use.")
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
    ))#[:99]

    # Load data
    mnist = MNIST()

    # Create the model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=[MNIST.H, MNIST.W, MNIST.C]),
        tf.keras.layers.Dense(args.hidden_layer, activation=tf.nn.relu),
        tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax),
    ])

    # : Use the required `args.optimizer` (either `SGD` or `Adam`).
    # For `SGD`, `args.momentum` can be specified.
    # - If `args.decay` is not specified, pass the given `args.learning_rate`
    #   directly to the optimizer as a `learning_rate` argument.
    # - If `args.decay` is set, then
    #   - for `linear`, use `tf.optimizers.schedules.PolynomialDecay` with defaul power=1.0
    #     using the given `args.learning_rate_final`;
    #     https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/PolynomialDecay
    #   - for `exponential`, use `tf.optimizers.schedules.ExponentialDecay`
    #     and set `decay_rate` appropriately to reach `args.learning_rate_final`
    #     just after the training (and keep the default `staircase=False`).
    #     https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay
    #   In both cases, `decay_steps` should be total number of training batches
    #   and you should pass the created `{Polynomial,Exponential}Decay` to
    #   the optimizer using the `learning_rate` constructor argument.
    #   The size of the training MNIST dataset is `mnist.train.size` and you
    #   can assume is it divisible by `args.batch_size`.
    #
    #   If a learning rate schedule is used, TensorBoard automatically logs the value of
    #   learning rate after every epoch. Additionally, you can find out the current learning
    #   rate manually by using `model.optimizer.learning_rate(model.optimizer.iterations)`,
    #   so after training this value should be `args.learning_rate_final`.

    # set learning rate
    if not args.decay:
        learning_rate = args.learning_rate
    else:
        decay_steps = (mnist.train.size / args.batch_size) * args.epochs
        if args.decay == 'linear':
            learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(decay_steps=decay_steps,
                                                                          initial_learning_rate=args.learning_rate,
                                                                          end_learning_rate=args.learning_rate_final,
                                                                          power=1.0)
        elif args.decay == 'exponential':
            decay_rate = args.learning_rate_final / args.learning_rate
            learning_rate = tf.optimizers.schedules.ExponentialDecay(decay_steps=decay_steps,
                                                                     decay_rate=decay_rate,
                                                                     initial_learning_rate=args.learning_rate)
            # print(learning_rate(decay_steps))
        else:
            raise NotImplementedError("Use only 'linear' or 'exponential' as LR scheduler")

    if args.optimizer == 'SGD':
        if args.momentum:
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=args.momentum)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif args.optimizer == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        raise NotImplementedError("Use only 'SGD' or 'Adam' as optimizer")

    model.compile(
        optimizer=optimizer,
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy("accuracy")],
    )

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

    def evaluate_test(epoch, logs):
        if epoch + 1 == args.epochs:
            test_logs = model.evaluate(
                mnist.test.data["images"], mnist.test.data["labels"], batch_size=args.batch_size, return_dict=True,
                verbose=0,
            )
            logs.update({"val_test_" + name: value for name, value in test_logs.items()})

    logs = model.fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=evaluate_test), tb_callback],
    )

    # Return test accuracy for ReCodEx to validate
    return logs.history["val_test_accuracy"][-1]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
