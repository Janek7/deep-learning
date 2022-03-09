#!/usr/bin/env python3
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
parser.add_argument("--batch_size", default=100, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


# The neural network model
class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        # Create a model with two inputs, both images of size [MNIST.H, MNIST.W, MNIST.C].
        images = (
            tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C]),
            tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C]),
        )

        # : The model starts by passing each input image through the same
        # subnetwork (with shared weights), which should perform
        # - convolution with 10 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation,
        # - convolution with 20 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation,
        # - flattening layer,
        # - fully connected layer with 200 neurons and ReLU activation,
        # obtaining a 200-dimensional feature representation FI of each image.
        conv_1 = tf.keras.layers.Conv2D(filters=10, kernel_size=3, strides=2, padding="valid", activation=tf.nn.relu)
        conv_2 = tf.keras.layers.Conv2D(filters=20, kernel_size=3, strides=2, padding="valid", activation=tf.nn.relu)
        flatten = tf.keras.layers.Flatten()
        hidden = tf.keras.layers.Dense(units=200, activation=tf.nn.relu)

        fi_1 = hidden(flatten(conv_2(conv_1(images[0]))))
        fi_2 = hidden(flatten(conv_2(conv_1(images[1]))))

        # : Using the computed representations, the model should produce four outputs:
        # - first, compute _direct prediction_ whether the first digit is
        #   greater than the second, by
        #   - concatenating the two 200-dimensional image representations FI,
        #   - processing them using another 200-neuron ReLU dense layer
        #   - computing one output using a dense layer with `tf.nn.sigmoid` activation
        concat = tf.keras.layers.concatenate([fi_1, fi_2])
        hidden2 = tf.keras.layers.Dense(units=200, activation=tf.nn.relu)
        direct_prediction_layer = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)
        direct_prediction = direct_prediction_layer(hidden2(concat))
        # - then, classify the computed representation FI of the first image using
        #   a densely connected softmax layer into 10 classes;
        # - then, classify the computed representation FI of the second image using
        #   the same connected layer (with shared weights) into 10 classes;
        classification = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
        digit_1 = classification(fi_1)
        digit_2 = classification(fi_2)
        # - finally, compute _indirect prediction_ whether the first digit
        #   is greater than second, by comparing the predictions from the above
        #   two outputs.
        outputs = {
            "direct_prediction": direct_prediction,
            "digit_1": digit_1,
            "digit_2": digit_2,
            # axis=1 is necessary for argmax because digit_1 is batch prediction
            "indirect_prediction": tf.cast(tf.math.argmax(digit_1, axis=1) > tf.math.argmax(digit_2, axis=1), tf.float32)
        }

        # Finally, construct the model.
        super().__init__(inputs=images, outputs=outputs)

        # Note that for historical reasons, names of a functional model outputs
        # (used for displayed losses/metric names) are derived from the name of
        # the last layer of the corresponding output. Here we instead use
        # the keys of the `outputs` dictionary.
        self.output_names = sorted(outputs.keys())

        # : Train the model by computing appropriate losses of
        # "direct_prediction", "digit_1", "digit_2". Regarding metrics, compute
        # the accuracy of both the direct and indirect predictions; name both
        # metrics "accuracy" (i.e., pass "accuracy" as the first argument of
        # the metric object).
        self.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss={  # keys fit to output dict
                "direct_prediction": tf.keras.losses.BinaryCrossentropy(),
                "digit_1": tf.keras.losses.SparseCategoricalCrossentropy(),
                "digit_2": tf.keras.losses.SparseCategoricalCrossentropy(),
            },
            metrics={
                "direct_prediction": [tf.keras.metrics.BinaryAccuracy("accuracy")],
                "indirect_prediction": [tf.keras.metrics.BinaryAccuracy("accuracy")]
            },
        )
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1)

    # Create an appropriate dataset using the MNIST data.
    def create_dataset(
        self, mnist_dataset: MNIST.Dataset, args: argparse.Namespace, training: bool = False
    ) -> tf.data.Dataset:
        # Start by using the original MNIST data
        dataset = tf.data.Dataset.from_tensor_slices((mnist_dataset.data["images"], mnist_dataset.data["labels"]))

        # : If `training`, shuffle the data with `buffer_size=10000` and `seed=args.seed`
        if training:
            dataset = dataset.shuffle(buffer_size=10000, seed=args.seed)

        # : Combine pairs of examples by creating batches of size 2
        dataset = dataset.batch(2)

        # : Map pairs of images to elements suitable for our model. Notably,
        # the elements should be pairs `(input, output)`, with
        # - `input` being a pair of images,
        # - `output` being a dictionary with keys "digit_1", "digit_2", "direct_prediction",
        #   and "indirect_prediction".
        def create_element(images, labels):
            output = {
                "digit_1": labels[0],
                "digit_2": labels[1],
                "direct_prediction": int(labels[0] > labels[1]),
                "indirect_prediction": int(labels[0] > labels[1])
            }
            return (images[0], images[1]), output

        dataset = dataset.map(create_element)

        # : Create batches of size `args.batch_size`
        dataset = dataset.batch(args.batch_size)

        return dataset


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
    ))

    # Load the data
    mnist = MNIST()

    # Create the model
    model = Model(args)

    # Construct suitable datasets from the MNIST data.
    train = model.create_dataset(mnist.train, args, training=True)
    dev = model.create_dataset(mnist.dev, args)
    test = model.create_dataset(mnist.test, args)

    # Train
    logs = model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[model.tb_callback])

    # Return development metrics for ReCodEx to validate
    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
