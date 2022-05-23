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

from omniglot_dataset import Omniglot

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--cell_size", default=40, type=int, help="Memory cell size")
parser.add_argument("--classes", default=5, type=int, help="Number of classes per episode.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--images_per_class", default=10, type=int, help="Images per class.")
parser.add_argument("--lstm_dim", default=256, type=int, help="LSTM Dim")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--read_heads", default=1, type=int, help="Read heads.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--test_episodes", default=1000, type=int, help="Number of testing episodes.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--train_episodes", default=10000, type=int, help="Number of training episodes.")
# If you add more arguments, ReCodEx will keep them with your default values.


class EpisodeGenerator():
    """Python generator of episodes."""
    def __init__(self, dataset, args, seed):
        self._dataset = dataset
        self._args = args

        # Random generator
        self._generator = np.random.RandomState(seed)

        # Create required indexes
        self._unique_labels = np.unique(dataset.data["labels"])
        self._label_indices = {}
        for i, label in enumerate(dataset.data["labels"]):
            self._label_indices.setdefault(label, []).append(i)

    def __call__(self):
        """Generate infinite number of episodes.

        Every episode contains `self._args.classes` randomly chosen Omniglot
        classes, each class being assigned a randomly chosen label. For every
        chosen class, `self._args.images_per_class` images are randomly selected.

        Apart from the images, the input contain the random labels one step
        after the corresponding images (with the first label being -1).
        The gold outputs are also the labels, but without the one-step offset.
        """
        while True:
            indices, labels = [], []
            for index, label in enumerate(self._generator.choice(
                    self._unique_labels, size=self._args.classes, replace=False)):
                indices.extend(self._generator.choice(
                    self._label_indices[label], size=self._args.images_per_class, replace=False))
                labels.extend([index] * self._args.images_per_class)
            indices, labels = np.array(indices, np.int32), np.array(labels, np.int32)

            permutation = self._generator.permutation(len(indices))
            images = self._dataset.data["images"][indices[permutation]]
            labels = labels[permutation]
            yield (images, np.pad(labels[:-1], [[1, 0]], constant_values=-1)), labels


class Model(tf.keras.Model):
    class NthOccurenceAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):
        """A sparse categorical accuracy computed only for `nth` occurrence of every element."""
        def __init__(self, nth, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._nth = nth

        def update_state(self, y_true, y_pred, sample_weight):
            assert sample_weight is None
            one_hot = tf.one_hot(y_true, tf.reduce_max(y_true) + 1)
            nth = tf.math.reduce_sum(tf.math.cumsum(one_hot, axis=-2) * one_hot, axis=-1)
            indices = tf.where(nth == self._nth)
            return super().update_state(tf.gather_nd(y_true, indices), tf.gather_nd(y_pred, indices))

    class MemoryAugmentedLSTM(tf.keras.layers.AbstractRNNCell):
        """The LSTM controller augmented with external memory.

        The LSTM has dimensionality `units`. The external memory consists
        of `memory_cells` cells, each being a vector of `cell_size` elements.
        The controller has `read_heads` read head and one write head.
        """
        def __init__(self, units, memory_cells, cell_size, read_heads, **kwargs):
            super().__init__(**kwargs)
            self._memory_cells = memory_cells
            self._cell_size = cell_size
            self._read_heads = read_heads

            # : Create the required layers:
            # - self._controller is a `tf.keras.layers.LSTMCell` with `units` units;
            # - self._parameters is a `tanh`-activated dense layer with `(read_heads + 1) * cell_size` units;
            # - self._output_layer is a `tanh`-activated dense layer with `units` units.
            self._controller = tf.keras.layers.LSTMCell(units)
            self._parameters = tf.keras.layers.Dense((read_heads + 1) * cell_size, activation=tf.nn.tanh)
            self._output_layer = tf.keras.layers.Dense(units, activation=tf.nn.tanh)

        @property
        def state_size(self):
            # TODO: Return the description of the state size as a list containing
            # sizes of individual state tensors. The state is a list consisting
            # of the following elements (in this order):
            # - first the state tensors of the `self._controller` itself; note that
            #   the `self._controller` also has `state_size` property;
            # - then the values of memory cells read by `self._read_heads` head
            #   in the previous time step;
            # - finally the external memory itself, which is a matrix containing
            #   `self._memory_cells` cells as rows, each of length `self._cell_size`.
            return [self._controller.state_size,
                    self._read_heads * self._cell_size,
                    tf.TensorShape([self._memory_cells, self._cell_size])]

        def call(self, inputs, states):
            # : Decompose `states` into `controller_state`, `read_value` and `memory`
            # (see `state_size` describing the `states` structure).
            controller_state, read_value, memory = states

            # : Call the LSTM controller, using a concatenation of `inputs` and
            # `read_value` (in this order) as input and `controller_state` as state.
            # Store the results in `controller_output` and `controller_state`.
            concatenated = tf.keras.layers.Concatenate()([inputs, read_value])
            controller_output, controller_state = self._controller(inputs=concatenated, states=controller_state)

            # : Pass the `controller_output` through the `self._parameters` layer, obtaining
            # the parameters for interacting with the external memory (in this order):
            # - `write_value` is the first `self._cell_size` elements of every batch example;
            # - `read_keys` is the rest of the elements of every batch example, reshaped to
            #   `[batch_size, self._read_heads, self._cell_size]`.
            hidden = self._parameters(controller_output)
            write_value = hidden[:, :self._cell_size]
            read_keys = tf.reshape(hidden[:, self._cell_size:], [args.batch_size, self._read_heads, self._cell_size])

            # : Read the memory. For every predicted read key, the goal is to
            # - compute cosine similarities between the key and all memory cells;
            # - compute cell distribution as a softmax of the computed cosine similarities;
            # - the read value is the sum of the memory cells weighted by the above distribution.
            #
            # However, implement the reading process in a vectorized way (for all read keys in parallel):
            # - compute L2 normalized copy of `memory` and `read_keys`, using `tf.math.l2_normalize`,
            #   so that every cell vector has norm 1;
            # - compute the self-attention between the L2-normalized copy of `read_keys` and `memory`
            #   with a single matrix multiplication, obtaining a tensor with shape
            #   `[batch_size, self._read_heads, self._memory_cells]`. You will need to transpose one
            #   of the matrices -- do not transpose it manually, but use `tf.linalg.matmul` capable of
            #   transposing the matrices to be multiplied (see parameters `transpose_a` and `transpose_b`).
            # - apply softmax, resulting in a distribution over the memory cells for every read key
            # - compute weighted sum of the original (non-L2-normalized) `memory` according to the
            #   obtained distribution. Compute it using a single matrix multiplication, producing
            #   a value with shape `[batch_size, self._read_heads, self._cell_size]`.
            # Finally, reshape the result into `read_value` of shape `[batch_size, self._read_heads * self._cell_size]`

            memory_L2 = tf.math.l2_normalize(memory, axis=2)
            read_keys_L2 = tf.math.l2_normalize(read_keys, axis=2)
            self_attention = tf.linalg.matmul(a=read_keys_L2, b=memory_L2, transpose_b=True)
            softmaxed = tf.nn.softmax(self_attention)
            weighted_sum = softmaxed @ memory
            read_value = tf.reshape(weighted_sum, [-1, self._read_heads * self._cell_size])

            # : Write to the memory by prepending the `write_value` as the first cell (row);
            # the last memory cell (row) is dropped.
            memory = tf.concat([write_value[:, tf.newaxis], memory[:, :-1]], axis=1)

            # : Generate `output` by concatenating `controller_output` and `read_value`
            # (in this order) and passing it through the `self._output_layer`.
            concatenated = tf.keras.layers.Concatenate()([controller_output, read_value])
            output = self._output_layer(concatenated)

            # : Return the `output` as output and a suitable combination of
            # `controller_state`, `read_value` and `memory` as state.
            return output, (controller_state, read_value, memory)

    def __init__(self, args):
        # Construct the model. The inputs are:
        # - a sequence of `images`;
        # - a sequence of labels of the previous images.
        images = tf.keras.layers.Input([None, Omniglot.H, Omniglot.W, Omniglot.C], dtype=tf.float32)
        previous_labels = tf.keras.layers.Input([None], dtype=tf.int32)

        # : Process each image with the same sequence of the following operations:
        # - convolutional layer with 8 filters, 3x3 kernel, stride 2, valid padding; BatchNorm; ReLU;
        # - convolutional layer with 16 filters, 3x3 kernel, stride 2, valid padding; BatchNorm; ReLU;
        # - convolutional layer with 32 filters, 3x3 kernel, stride 2, valid padding; BatchNorm; ReLU;
        # - finally, flatten each image into a vector.

        bn_relu = lambda input: tf.keras.layers.ReLU()(tf.keras.layers.BatchNormalization()(input))
        conv1 = bn_relu(tf.keras.layers.Conv2D(8, 3, 2, "valid")(images))
        conv2 = bn_relu(tf.keras.layers.Conv2D(16, 3, 2, "valid")(conv1))
        conv3 = bn_relu(tf.keras.layers.Conv2D(32, 3, 2, "valid")(conv2))
        # reshape to sequence of images, multiplication of w*h*c
        flattened = tf.keras.layers.Reshape([-1, np.prod(conv3.shape[-3:])])(conv3)

        # : To create the input for the MemoryAugmentedLSTM, concatenate (in this order)
        # each computed image representation with the one-hot representation of the
        # label of the previous image from `previous_labels`.
        concatenated = tf.keras.layers.Concatenate()([flattened, tf.one_hot(previous_labels, depth=args.classes)])

        # : Create the MemoryAugmentedLSTM cell, using
        # - `args.lstm_dim` units;
        # - `args.classes * args.images_per_class` memory cells of size `args.cell_size`;
        # - `args.read_heads` read heads.
        # Then, run this cell using `tf.keras.layers.RNN` on the prepared input,
        # obtaining output for every input sequence element.
        cell = Model.MemoryAugmentedLSTM(units=args.lstm_dim, memory_cells=args.classes * args.images_per_class,
                                         cell_size=args.cell_size, read_heads=args.read_heads)
        rnn_output = tf.keras.layers.RNN(cell, return_sequences=True)(concatenated)

        # : Pass the sequence of outputs through a classification dense layer
        # with `args.classes` units and `tf.nn.softmax` activation.
        predictions = tf.keras.layers.Dense(args.classes, activation=tf.nn.softmax)(rnn_output)

        # Create the model and compile it.
        super().__init__(inputs=[images, previous_labels], outputs=predictions)
        self.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="acc"),
                     *[self.NthOccurenceAccuracy(i, name="acc{}".format(i)) for i in [1, 2, 5, 10]]],
        )


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

    # Create the data
    omniglot = Omniglot()
    def create_dataset(data, seed):
        dataset = tf.data.Dataset.from_generator(
            EpisodeGenerator(data, args, seed=seed),
            output_signature=(
                (tf.TensorSpec([None, Omniglot.H, Omniglot.W, Omniglot.C], tf.float32),
                 tf.TensorSpec([None], tf.int32)),
                tf.TensorSpec([None], tf.int32),
            )
        )
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(tf.data.INFINITE_CARDINALITY))
        return dataset
    train = create_dataset(omniglot.train, args.seed).take(args.train_episodes).batch(args.batch_size).prefetch(1)
    test = create_dataset(omniglot.test, seed=42).take(args.test_episodes).batch(args.batch_size).cache()

    # Create the model and train
    model = Model(args)
    logs = model.fit(train, epochs=args.epochs, validation_data=test)

    # Return development and training losses for ReCodEx to validate
    return {metric: values[-1] for metric, values in logs.history.items() if "loss" in metric}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
