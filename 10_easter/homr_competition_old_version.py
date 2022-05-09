#!/usr/bin/env python3
# TEAM MEMBERS:
# Antonio Krizmanic - 2b193238-8e3c-11ec-986f-f39926f24a9c
# Janek Putz - e31a3cae-8e6c-11ec-986f-f39926f24a9c
import argparse
import datetime
import functools
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from homr_dataset import HOMRDataset

# : Define reasonable defaults and optionally more parameters.
# Also, you can set the number of the threads 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--weight_path", default=None, type=str, help="Path to restore weights")
# architecture params
parser.add_argument("--image_height", default=50, type=int, help="Height to resize images")
parser.add_argument("--image_width", default=500, type=int, help="Width to resize images")
parser.add_argument("--cnn_model", default="normal", type=str, help="Model type (normal/resnet/widenet")
parser.add_argument("--cnn_filters", default=32, type=int, help="Normal CNN variant #filters")
parser.add_argument("--cnn_layers", default=3, type=int, help="Normal CNN variant #layers")
parser.add_argument("--depth", default=56, type=int, help="CNN model depth")
parser.add_argument("--width", default=1, type=int, help="WideNet model width")
parser.add_argument("--conv_slices", default=25, type=int, help="Slices of conv representations as input for RNN")
parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
parser.add_argument("--rnn_cell_dim", default=64, type=int, help="RNN cell dimension.")
parser.add_argument("--rnn_layers", default=1, type=int, help="Number of bidirectional rnn layers behind each other.")
parser.add_argument("--dense_layers", default=None, type=str, help="List of dense layers (specified by units, ',' separated)")
# regularization params
parser.add_argument("--dropout", default=0, type=float, help="Dropout rate.")
parser.add_argument("--label_smoothing", default=0.0, type=float, help="Label smoothing.")
parser.add_argument("--decay", default=None, type=str, help="Learning decay rate type")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default=0.0001, type=float, help="Final learning rate.")
parser.add_argument("--l2", default=0.00, type=float, help="L2 regularization.")


# A layer retrieving fast text vectors for word.
class SliceConv(tf.keras.layers.Layer):
    def __init__(self, number_slices):
        super().__init__()
        self._number_slices = number_slices

    def get_config(self):
        return {"number_slices": self._number_slices}

    def call(self, inputs, training):
        # prepare variables
        sequence_length = tf.cast(tf.shape(inputs)[2], tf.int32)  # width
        slice_length = tf.cast(sequence_length / self._number_slices, tf.int32)
        # assert self._number_slices * slice_length == sequence_length, \
        #     f"number of slices {self._number_slices} does not divide sequence length {sequence_length} without rest"

        # slicing and flattening
        flatten = tf.keras.layers.Flatten()
        l = []
        for i in range(self._number_slices):
            slice = inputs[
                    :,  # batches
                    :,  # keep height
                    i * slice_length: (i + 1) * slice_length,  # slice width
                    :  # keep all channels
                    ]
            slice_flattened = flatten(slice)
            l.append(tf.expand_dims(slice_flattened, 1))

        return tf.concat(l, axis=1)


class ComplexConvLayer(tf.keras.layers.Layer):
    # abstract constructor and config from `ResNetLayer` and `WideNetLayer`
    def __init__(self, args):
        super().__init__()
        self._n = (args.depth - 2) // 6

    def get_config(self):
        return {"n": self._n}


# Layers that encapsulates ResNet convolutions (without global average pooling and head)
class ResNetLayer(ComplexConvLayer):

    def _cnn(self, inputs, filters, kernel_size, stride, activation):
        hidden = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding="same", use_bias=False)(inputs)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.ReLU()(hidden) if activation else hidden
        return hidden

    def _block(self, inputs, filters, stride):
        hidden = self._cnn(inputs, filters, 3, stride, activation=True)
        hidden = self._cnn(hidden, filters, 3, 1, activation=False)
        if stride > 1:
            residual = self._cnn(inputs, filters, 1, stride, activation=False)
        else:
            residual = inputs
        hidden = tf.keras.layers.ReLU()(hidden + residual)
        return hidden

    def call(self, inputs, training):
        hidden = self._cnn(inputs, 16, 3, 1, activation=True)
        for stage in range(3):
            for block in range(self._n):
                hidden = self._block(hidden, 16 * (1 << stage), 2 if stage > 0 and block == 0 else 1)
        return hidden


# Layers that encapsulates WideNet convolutions (without global average pooling and head)
class WideNetLayer(ComplexConvLayer):

    def _cnn(self, inputs, filters, kernel_size, stride):
        return tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding="same", use_bias=False)(inputs)

    def _bn_activation(self, inputs):
        return tf.keras.layers.ReLU()(tf.keras.layers.BatchNormalization()(inputs))

    def _block(self, inputs, filters, stride):
        hidden = self._bn_activation(inputs)
        hidden = self._cnn(hidden, filters, 3, stride)
        hidden = self._bn_activation(hidden)
        hidden = tf.keras.layers.Dropout(args.dropout)(hidden)
        hidden = self._cnn(hidden, filters, 3, 1)
        if stride > 1:
            residual = self._cnn(inputs, filters, 1, stride)
        else:
            residual = inputs
        return hidden + residual

    def call(self, inputs, training):
        hidden = self._cnn(inputs, 16, 3, 1)
        for stage in range(3):
            for block in range(self._n):
                hidden = self._block(hidden, 16 * args.width * (1 << stage), 2 if block == 0 else 1)
        hidden = self._bn_activation(hidden)
        return hidden


# Model
class Model(tf.keras.Model):

    def __init__(self, args: argparse.Namespace, train: tf.data.Dataset) -> None:
        print(" Model ".center(50, '*'))
        # A) REGULARIZATION
        if not args.decay or args.decay in ["None", "none"]:
            learning_rate = args.learning_rate
        else:
            # note: train is already batched
            decay_steps = len(train) * args.epochs
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
            elif args.decay == 'cosine':
                learning_rate = tf.keras.optimizers.schedules.CosineDecay(decay_steps=decay_steps,
                                                                          initial_learning_rate=args.learning_rate)
            else:
                raise NotImplementedError("Use only 'linear', 'exponential' or 'cosine' as LR scheduler")

        # B) ARCHITECTURE
        inputs = tf.keras.Input(shape=(args.image_height, args.image_width, 1))
        print("inputs", inputs.shape)

        # CNN (use of ResNet and WideNet only internal representation without global avg pooling  & class. header)
        if args.cnn_model == 'normal':
            conv = inputs
            for i in range(args.cnn_layers):
                conv = self.bn_relu(tf.keras.layers.Conv2D(args.cnn_filters, 3, 1, "same")(conv))
        elif args.cnn_model == 'resnet':
            conv = ResNetLayer(args)(inputs)
        elif args.cnn_model == 'widenet':
            conv = WideNetLayer(args)(inputs)
        else:
            raise ValueError(f"cnn model must be 'normal/resnet/widenet' and not {args.cnn_model}")
        print("conv representation", conv.shape)
        # slice and flatten conv as input for rnn
        flattened_slices = SliceConv(number_slices=args.conv_slices)(conv)
        print("flattened", flattened_slices.shape)

        # RNN
        if args.rnn_cell == 'LSTM':
            cell_type = tf.keras.layers.LSTM
        elif args.rnn_cell == 'GRU':
            cell_type = tf.keras.layers.GRU
        else:
            raise NotImplementedError(f"{args.rnn_cell} is not a valid RNN cell")

        rnn_sequences = flattened_slices
        rnn_sequences_previous = None
        for i in range(args.rnn_layers):
            rnn_sequences = tf.keras.layers.Bidirectional(cell_type(units=args.rnn_cell_dim, return_sequences=True),
                                                          merge_mode="sum")(rnn_sequences)
            rnn_sequences = tf.keras.layers.Dropout(args.dropout)(rnn_sequences)
            if rnn_sequences_previous is not None:
                rnn_sequences = tf.keras.layers.Add(name=f"rnn_residual_connection_{i}")(
                    [rnn_sequences_previous, rnn_sequences])
            # set previous sequence for next loop
            rnn_sequences_previous = rnn_sequences
        print("rnn", rnn_sequences.shape)
        dropout = tf.keras.layers.Dropout(args.dropout)(rnn_sequences)

        # DENSE LAYER
        dense = dropout
        if args.dense_layers:
            for units in args.dense_layers.split(","):
                dense = tf.keras.layers.Dense(int(units), activation=tf.nn.relu)(dense)
                dense = tf.keras.layers.Dropout(args.dropout)(dense)

        # OUTPUT
        outputs = tf.keras.layers.Dense(1 + len(HOMRDataset.MARKS), activation=None)(dense)
        print("output", outputs.shape)
        print("".center(50, '*'))
        # keep all values, just transform so it fits to ctc_loss
        outputs = tf.RaggedTensor.from_tensor(outputs, padding=None)

        super().__init__(inputs=inputs, outputs=outputs)

        # C) COMPILE

        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=self.ctc_loss,
            metrics=[HOMRDataset.EditDistanceMetric("edit_distance")]
        )

    # the following five methods are the same as in speech recognition

    def ctc_loss(self, gold_labels: tf.RaggedTensor, logits: tf.RaggedTensor) -> tf.Tensor:
        assert isinstance(gold_labels, tf.RaggedTensor), "Gold labels given to CTC loss must be RaggedTensors"
        assert isinstance(logits, tf.RaggedTensor), "Logits given to CTC loss must be RaggedTensors"

        # : Use tf.nn.ctc_loss to compute the CTC loss.
        # - Convert the gold_labels to SparseTensor and pass `None` as `label_length`.
        # - Convert `logits` to a dense Tensor and then either transpose the
        #   logits to `[max_audio_length, batch, dim]` or set `logits_time_major=False`
        # - Use `logits.row_lengths()` method to obtain the `logit_length`
        # - Use the last class (the one with the highest index) as the `blank_index`.
        #
        # The `tc.nn.ctc_loss` returns a value for a single batch example, so average
        # them to produce a single value and return it.
        gold_labels_sparse = tf.cast(gold_labels.to_sparse(), tf.int32)
        logits_dense = logits.to_tensor()
        logit_length = tf.cast(logits.row_lengths(), tf.int32)
        # tf.print(tf.shape(gold_labels_sparse))
        # tf.print(tf.shape(logits_dense))
        result = tf.nn.ctc_loss(labels=gold_labels_sparse, logits=logits_dense,
                                label_length=None, logit_length=logit_length,
                                logits_time_major=False,
                                blank_index=len(HOMRDataset.MARKS))
        avg_result = tf.math.reduce_mean(result)
        return avg_result

    def ctc_decode(self, logits: tf.RaggedTensor) -> tf.RaggedTensor:
        assert isinstance(logits, tf.RaggedTensor), "Logits given to CTC predict must be RaggedTensors"

        # : Run `tf.nn.ctc_greedy_decoder` or `tf.nn.ctc_beam_search_decoder`
        # to perform prediction.
        # - Convert the `logits` to a dense Tensor and then transpose them
        #   to shape `[max_audio_length, batch, dim]` using `tf.transpose`
        # - Use `logits.row_lengths()` method to obtain the `sequence_length`
        # - Convert the result of the decoded from a SparseTensor to a RaggedTensor
        decoded, _ = tf.nn.ctc_greedy_decoder(inputs=tf.transpose(logits.to_tensor(), [1, 0, 2]),
                                              sequence_length=tf.cast(logits.row_lengths(), tf.int32))
        predictions = tf.RaggedTensor.from_sparse(decoded[0])
        assert isinstance(predictions, tf.RaggedTensor), "CTC predictions must be RaggedTensors"
        return predictions

    # We override the `train_step` method, because we do not want to
    # evaluate the training data for performance reasons
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return {"loss": metric.result() for metric in self.metrics if metric.name == "loss"}

    # We override `predict_step` to run CTC decoding during prediction.
    def predict_step(self, data):
        data = data[0] if isinstance(data, tuple) else data
        y_pred = self(data, training=False)
        y_pred = self.ctc_decode(y_pred)
        return y_pred

    # We override `test_step` to run CTC decoding during evaluation.
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compute_loss(x, y, y_pred)
        y_pred = self.ctc_decode(y_pred)
        return self.compute_metrics(x, y, y_pred, None)

    @staticmethod
    def bn_relu(input):
        return tf.keras.layers.ReLU()(tf.keras.layers.BatchNormalization()(input))


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

    # Load the data
    homr = HOMRDataset()

    def create_dataset(name):
        dataset = getattr(homr, name)

        def prepare_data(example):
            # resize image with keeping aspect ratios and pad with 0s if aspect ratio does not fit to target dimensions
            image_padded = tf.image.resize_with_pad(example["image"],
                                                    target_height=args.image_height, target_width=args.image_width)
            # create ragged tensor not possible because Conv needs a normal tensor
            # image_ragged = tf.RaggedTensor.from_tensor(image_padded, padding=None)
            return image_padded, example["marks"]

        dataset = dataset.map(prepare_data)
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        # dataset = dataset.batch(args.batch_size)  # does not work because of different number of outputs
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        return dataset

    train, dev, test = create_dataset("train"), create_dataset("dev"), create_dataset("test")
    print("Data Structure:")
    for b in dev.take(1):
        print("X", b[0].shape, type(b[0]))
        print("y", b[1].shape, type(b[1]))

    # : Create the model and train it
    model = Model(args, train)

    if args.weight_path:
        print(f"load model from {args.weight_path}")
        model.load_weights(args.weight_path)
    else:
        print("create new model")

    print(args)
    model.fit(
        train.take(1), batch_size=args.batch_size, epochs=args.epochs, validation_data=dev.take(1),
        callbacks=[tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0),
                   tf.keras.callbacks.EarlyStopping(monitor='val_edit_distance', min_delta=1e-4, patience=10,
                                                    verbose=0, mode="min", baseline=None, restore_best_weights=True)]
    )
    print(args)

    weights_path = os.path.join(args.logdir, "weights")
    model.save_weights(weights_path)
    print(f"saved weights as '{weights_path}'")

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "homr_competition.txt"), "w", encoding="utf-8") as predictions_file:
        print("Create predictions on test set")
        predictions = model.predict(test)
        for sequence in predictions:
            print(" ".join(homr.MARKS[mark] for mark in sequence), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
