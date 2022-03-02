#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layers", default=[100], nargs="*", type=int, help="Hidden layer sizes.")
parser.add_argument("--model_type", default="sequential", type=str, help="Model type (sequential, functional, subclassing)")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load data
    mnist = MNIST()

    # Create the model
    if args.model_type == "sequential":
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input([MNIST.H, MNIST.W, MNIST.C]))
        model.add(tf.keras.layers.Flatten())
        for hidden_layer in args.hidden_layers:
            model.add(tf.keras.layers.Dense(hidden_layer, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax))
        model.summary()

    elif args.model_type == "functional":
        inputs = tf.keras.layers.Input([MNIST.H, MNIST.W, MNIST.C])
        hidden = tf.keras.layers.Flatten()(inputs)
        for hidden_layer in args.hidden_layers:
            hidden = tf.keras.layers.Dense(hidden_layer, activation=tf.nn.relu)(hidden)
        outputs = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)(hidden)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.summary()

    elif args.model_type == "subclassing":
        class Model(tf.keras.Model):
            def __init__(self, hidden_layers):
                super().__init__()

                self.flatten_layer = tf.keras.layers.Flatten()
                self.hidden_layers = [tf.keras.layers.Dense(hidden_layer, activation=tf.nn.relu) for hidden_layer in hidden_layers]
                self.output_layer = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)

            def call(self, inputs):
                hidden = self.flatten_layer(inputs)
                for hidden_layer in self.hidden_layers:
                    hidden = hidden_layer(hidden)
                return self.output_layer(hidden)

        model = Model(args.hidden_layers)

    else:
        raise ValueError("Unknown model type '{}'".format(args.model_type))

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy("accuracy")],
    )

    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

    def evaluate_test(epoch, logs):
        if epoch + 1 == args.epochs:
            test_logs = model.evaluate(
                mnist.test.data["images"], mnist.test.data["labels"], batch_size=args.batch_size, return_dict=True, verbose=0,
            )
            logs.update({"val_test_" + name: value for name, value in test_logs.items()})

    model.fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=evaluate_test), tb_callback],
    )

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
