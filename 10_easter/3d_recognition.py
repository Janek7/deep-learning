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
import tensorflow_addons as tfa

from modelnet import ModelNet

# : Define reasonable defaults and optionally more parameters.
# Also, you can set the number of the threads 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--modelnet", default=20, type=int, help="ModelNet dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

parser.add_argument("--model", default="resnet", type=str, help="Model type (resnet/widenet")
parser.add_argument("--activation", default="relu", type=str, help="Activation type")
parser.add_argument("--decay", default="cosine", type=str, help="Decay type")
parser.add_argument("--depth", default=56, type=int, help="Model depth")
parser.add_argument("--dropout", default=0., type=float, help="Dropout")
parser.add_argument("--label_smoothing", default=0., type=float, help="Label smoothing.")
parser.add_argument("--learning_rate", default=None, type=float, help="Learning rate")
parser.add_argument("--optimizer", default="SGD", type=str, help="Optimizer type")
parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay")
parser.add_argument("--width", default=1, type=int, help="Model width")


class Model(tf.keras.Model):
    def _activation(self, inputs, args):
        if args.activation == "relu":
            return tf.keras.layers.Activation(tf.nn.relu)(inputs)
        if args.activation == "lrelu":
            return tf.keras.layers.Activation(tf.nn.leaky_relu)(inputs)
        if args.activation == "elu":
            return tf.keras.layers.Activation(tf.nn.elu)(inputs)
        if args.activation == "swish":
            return tf.keras.layers.Activation(tf.nn.swish)(inputs)
        if args.activation == "gelu":
            return tf.keras.layers.Activation(tf.nn.gelu)(inputs)
        raise ValueError("Unknown activation '{}'".format(args.activation))


class ResNet3D(Model):
    def _cnn(self, inputs, args, filters, kernel_size, stride, activation):
        hidden = tf.keras.layers.Conv3D(filters, kernel_size=kernel_size, strides=stride, padding="same", use_bias=False)(inputs)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = self._activation(hidden, args) if activation else hidden
        return hidden

    def _block(self, inputs, args, filters, stride):
        hidden = self._cnn(inputs, args, filters, 3, stride, activation=True)
        hidden = self._cnn(hidden, args, filters, 3, 1, activation=False)
        if stride > 1:
            residual = self._cnn(inputs, args, filters, 1, stride, activation=False)
        else:
            residual = inputs
        hidden = self._activation(hidden + residual, args)
        return hidden

    def __init__(self, args, modelnet: ModelNet):
        n = (args.depth - 2) // 6

        inputs = tf.keras.Input(shape=[modelnet.D, modelnet.H, modelnet.W, modelnet.C], dtype=tf.float32)
        hidden = self._cnn(inputs, args, 16, 3, 1, activation=True)
        for stage in range(3):
            for block in range(n):
                hidden = self._block(hidden, args, 16 * (1 << stage), 2 if stage > 0 and block == 0 else 1)
        hidden = tf.keras.layers.GlobalAvgPool3D()(hidden)
        outputs = tf.keras.layers.Dense(len(modelnet.LABELS), activation=tf.nn.softmax)(hidden)
        super().__init__(inputs, outputs)


class WideNet3D(Model):
    def _cnn(self, inputs, args, filters, kernel_size, stride):
        return tf.keras.layers.Conv3D(filters, kernel_size=kernel_size, strides=stride, padding="same", use_bias=False)(inputs)

    def _bn_activation(self, inputs, args):
        return self._activation(tf.keras.layers.BatchNormalization()(inputs), args)

    def _block(self, inputs, args, filters, stride):
        hidden = self._bn_activation(inputs, args)
        hidden = self._cnn(hidden, args, filters, 3, stride)
        hidden = self._bn_activation(hidden, args)
        hidden = tf.keras.layers.Dropout(args.dropout)(hidden)
        hidden = self._cnn(hidden, args, filters, 3, 1)
        if stride > 1:
            residual = self._cnn(inputs, args, filters, 1, stride)
        else:
            residual = inputs
        return hidden + residual

    def __init__(self, args, modelnet: ModelNet):
        n = (args.depth - 4) // 6

        inputs = tf.keras.Input(shape=[modelnet.D, modelnet.H, modelnet.W, modelnet.C], dtype=tf.float32)
        hidden = self._cnn(inputs, args, 16, 3, 1)
        for stage in range(3):
            for block in range(n):
                hidden = self._block(hidden, args, 16 * args.width * (1 << stage), 2 if block == 0 else 1)
        hidden = self._bn_activation(hidden, args)
        hidden = tf.keras.layers.GlobalAvgPool3D()(hidden)
        outputs = tf.keras.layers.Dense(len(modelnet.LABELS), activation=tf.nn.softmax)(hidden)
        super().__init__(inputs, outputs)


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
    modelnet = ModelNet(args.modelnet)
    # transform labels to categorical for label smoothing
    for dataset in [modelnet.train, modelnet.dev]: # , modelnet.test]:
        dataset.data["labels"] = tf.keras.utils.to_categorical(dataset.data["labels"], num_classes=len(ModelNet.LABELS))

    def create_dataset(dataset: tf.data.Dataset, training: bool) -> tf.data.Dataset:
        def prepare_example(example):
            return example["voxels"], example["labels"]

        dataset = dataset.map(prepare_example)
        if training:
            dataset = dataset.shuffle(buffer_size=10000, seed=args.seed)
        dataset = dataset.batch(args.batch_size)
        return dataset

    train = create_dataset(modelnet.train.dataset, True)
    dev = create_dataset(modelnet.dev.dataset, False)
    test = create_dataset(modelnet.test.dataset, False)

    # Decay
    training_batches = args.epochs * modelnet.train.size // args.batch_size
    if args.decay == "piecewise":
        decay_fn = lambda value: tf.optimizers.schedules.PiecewiseConstantDecay(
            [int(0.5 * training_batches), int(0.75 * training_batches)],
            [value, value / 10, value / 100])
    elif args.decay == "cosine":
        decay_fn = lambda value: tf.keras.experimental.CosineDecay(value, training_batches)
    else:
        raise ValueError("Uknown decay '{}'".format(args.decay))
    if args.learning_rate is None:
        args.learning_rate = 0.1 if args.optimizer == "SGD" else 0.01
    learning_rate = decay_fn(args.learning_rate)
    weight_decay = decay_fn(args.weight_decay)

    # Optimizer
    if args.optimizer == "SGD":
        optimizer = tfa.optimizers.SGDW(learning_rate=learning_rate, weight_decay=weight_decay, momentum=0.9,
                                        nesterov=True)
    elif args.optimizer == "RMSProp":
        optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay, beta_2=0.9,
                                         epsilon=1e-3)
    elif args.optimizer.startswith("Adam"):
        beta2, epsilon = map(float, args.optimizer.split(":")[1:])
        optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay, beta_2=beta2,
                                         epsilon=epsilon)
    else:
        raise ValueError("Uknown optimizer '{}'".format(args.optimizer))

    # create model
    if args.model == "resnet":
        model = ResNet3D(args, modelnet)
    elif args.model == "widenet":
        model = WideNet3D(args, modelnet)
    else:
        raise ValueError("Unknown model '{}'".format(args.model))
    model.compile(
        optimizer=optimizer,
        loss=tf.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
        metrics=[tf.metrics.CategoricalAccuracy("accuracy")],
    )

    # Train
    print(args)
    best_checkpoint_path = os.path.join(args.logdir, "3d_recognition.ckpt")
    model.fit(
        train.take(1), batch_size=args.batch_size, epochs=args.epochs, validation_data=dev.take(1),
        callbacks=[tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0),
                   tf.keras.callbacks.ModelCheckpoint(filepath=best_checkpoint_path,
                                                      save_weights_only=True, monitor='val_accuracy',
                                                      mode='max', save_best_only=True)]
    )
    print(args)
    model.load_weights(best_checkpoint_path)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "3d_recognition.txt"), "w", encoding="utf-8") as predictions_file:
        # : Predict the probabilities on the test set
        print("create test set predictions")
        test_probabilities = model.predict(test.take(1))
        for probs in test_probabilities:
            print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
