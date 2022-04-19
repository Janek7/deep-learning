#!/usr/bin/env python3
import argparse
import datetime
import os
import re
import sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import tensorflow_addons as tfa

import autoaugment
from cifar10 import CIFAR10

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--activation", default="relu", type=str, help="Activation type")
parser.add_argument("--augmentation", default="basic", type=str, help="Augmentation type")
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--decay", default="piecewise", type=str, help="Decay type")
parser.add_argument("--depth", default=56, type=int, help="Model depth")
parser.add_argument("--dropout", default=0., type=float, help="Dropout")
parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
parser.add_argument("--label_smoothing", default=0., type=float, help="Label smoothing.")
parser.add_argument("--learning_rate", default=None, type=float, help="Learning rate")
parser.add_argument("--model", default="resnet", type=str, help="Model type (resnet/widenet")
parser.add_argument("--optimizer", default="SGD", type=str, help="Optimizer type")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
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


class ResNet(Model):
    def _cnn(self, inputs, args, filters, kernel_size, stride, activation):
        hidden = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding="same", use_bias=False)(inputs)
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

    def __init__(self, args):
        n = (args.depth - 2) // 6

        inputs = tf.keras.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C], dtype=tf.float32)
        hidden = self._cnn(inputs, args, 16, 3, 1, activation=True)
        for stage in range(3):
            for block in range(n):
                hidden = self._block(hidden, args, 16 * (1 << stage), 2 if stage > 0 and block == 0 else 1)
        hidden = tf.keras.layers.GlobalAvgPool2D()(hidden)
        outputs = tf.keras.layers.Dense(CIFAR10.LABELS, activation=tf.nn.softmax)(hidden)
        super().__init__(inputs, outputs)


class WideNet(Model):
    def _cnn(self, inputs, args, filters, kernel_size, stride):
        return tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding="same", use_bias=False)(inputs)

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

    def __init__(self, args):
        n = (args.depth - 4) // 6

        inputs = tf.keras.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C], dtype=tf.float32)
        hidden = self._cnn(inputs, args, 16, 3, 1)
        for stage in range(3):
            for block in range(n):
                hidden = self._block(hidden, args, 16 * args.width * (1 << stage), 2 if block == 0 else 1)
        hidden = self._bn_activation(hidden, args)
        hidden = tf.keras.layers.GlobalAvgPool2D()(hidden)
        outputs = tf.keras.layers.Dense(CIFAR10.LABELS, activation=tf.nn.softmax)(hidden)
        super().__init__(inputs, outputs)


def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "cifar", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load data
    cifar = CIFAR10()

    # TODO: Create the model and train it
    for dataset in [cifar.train, cifar.dev, cifar.test]:
        dataset.data["labels"] = tf.keras.utils.to_categorical(dataset.data["labels"], num_classes=CIFAR10.LABELS)

    # Model
    if args.model == "resnet":
        model = ResNet(args)
    elif args.model == "widenet":
        model = WideNet(args)
    else:
        raise ValueError("Unknown model '{}'".format(args.model))

    # Decay
    training_batches = args.epochs * cifar.train.size // args.batch_size
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
        optimizer = tfa.optimizers.SGDW(learning_rate=learning_rate, weight_decay=weight_decay, momentum=0.9, nesterov=True)
    elif args.optimizer == "RMSProp":
        optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay, beta_2=0.9, epsilon=1e-3)
    elif args.optimizer.startswith("Adam"):
        beta2, epsilon = map(float, args.optimizer.split(":")[1:])
        optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay, beta_2=beta2, epsilon=epsilon)
    else:
        raise ValueError("Uknown optimizer '{}'".format(args.optimizer))

    # Augmentation
    def cutout(image):
        y, x = np.random.randint(image.shape[0]), np.random.randint(image.shape[1])
        image = image.copy()
        image[max(0, y - 8):y + 8, max(0, x - 8):x + 8] = 127 / 255
        return image

    tf.get_logger().setLevel("ERROR") # Do not log warnings about multiprocessing
    if args.augmentation == "none":
        generator_opts = {}
    elif args.augmentation in ["basic", "cutout", "autoaugment"]:
        generator_opts = {
            "horizontal_flip": True,
            "width_shift_range": 4 + 1,
            "height_shift_range": 4 + 1,
            "fill_mode": "constant",
            "cval": 127 / 255,
        }
        if args.augmentation == "cutout":
            generator_opts["preprocessing_function"] = cutout
        if args.augmentation == "autoaugment":
            autoaugment_policy = autoaugment.CIFAR10Policy(fillcolor=(127, 127, 127))
            def autoaugmenter(image):
                import PIL.Image
                return np.array(autoaugment_policy(PIL.Image.fromarray((image*255).astype(np.uint8))), dtype=np.float32) / 255
            generator_opts["preprocessing_function"] = lambda image: cutout(autoaugmenter(image))
    else:
        raise ValueError("Uknown augmentation '{}'".format(args.augmentation))
    generator = tf.keras.preprocessing.image.ImageDataGenerator(**generator_opts)

    model.compile(
        optimizer=optimizer,
        loss=tf.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
        metrics=[tf.metrics.CategoricalAccuracy("accuracy")],
    )

    hparams = {name: value for name, value in vars(args).items() if name not in ["logdir", "seed", "threads"]}
    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)
    tb_callback._close_writers = lambda: None  # A hack allowing to keep the writers open.

    os.makedirs(args.logdir)
    with open(os.path.join(args.logdir, "cifar_competition.log"), "w", encoding="utf-8") as log_file:
        sys.stdout = log_file

        logs = model.fit(
            generator.flow(cifar.train.data["images"], cifar.train.data["labels"], batch_size=args.batch_size, seed=args.seed),
            epochs=args.epochs, use_multiprocessing=True, workers=args.threads,
            validation_data=(cifar.dev.data["images"], cifar.dev.data["labels"]),
            callbacks=[tb_callback, hp.KerasCallback(args.logdir, hparams)], verbose=2,
        )

        test_logs = model.evaluate(
            cifar.test.data["images"], cifar.test.data["labels"], batch_size=args.batch_size, return_dict=True,
        )
        tb_callback.on_epoch_end(args.epochs, {"val_test_" + metric: value for metric, value in test_logs.items()})

        print(*("{}={}".format(key, value) for key, value in sorted(vars(args).items())),
              "dev={}".format(100 * logs.history["val_accuracy"][-1]),
              "test={}".format(100 * test_logs["accuracy"]),
              file=log_file)

        model.save(os.path.join(args.logdir, "model.h5"), include_optimizer=False)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as predictions_file:
        for probs in model.predict(cifar.test.data["images"], batch_size=args.batch_size):
            print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)