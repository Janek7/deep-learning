#!/usr/bin/env python3
# TEAM MEMBERS:
# Antonio Krizmanic - 2b193238-8e3c-11ec-986f-f39926f24a9c
# Janek Putz - e31a3cae-8e6c-11ec-986f-f39926f24a9c

import argparse
import datetime
import os
import re
from typing import Dict, Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
try:
    import transformers
except Exception:
    raise RuntimeError("You need to install the `transformers` package")

from text_classification_dataset import TextClassificationDataset

# : Define reasonable defaults and optionally more parameters.
# Also, you can set the number of the threads 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
# Standard params
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# Architecture params
parser.add_argument("--dropout", default=0., type=float, help="Dropout")
parser.add_argument("--decay", default="None", type=str, help="Learning decay rate type")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default=0.0001, type=float, help="Final learning rate.")

# use 0.1 dropout and learning rate 5e-5

class LinearWarmup(tf.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps, following_schedule):
        self._warmup_steps = warmup_steps
        self._warmup = tf.optimizers.schedules.PolynomialDecay(0., warmup_steps, following_schedule(0))
        self._following = following_schedule

    def __call__(self, step):
        return tf.cond(step < self._warmup_steps,
                       lambda: self._warmup(step),
                       lambda: self._following(step - self._warmup_steps))


class Model(tf.keras.Model):
    def __init__(self, args, eleczech, train):

        # A) REGULARIZATION
        decay_steps = len(train) * args.epochs
        if not args.decay or args.decay in ["None", "none"]:
            # constant rate wrapped in callable schedule for warmup steps later
            learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(decay_steps=decay_steps,
                                                                          initial_learning_rate=args.learning_rate,
                                                                          end_learning_rate=args.learning_rate,
                                                                          power=1.0)
        else:
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

        # create warmup of one epoch
        warmup_steps = len(train)  # len(train) -> number of steps in one epoch
        learning_rate = LinearWarmup(warmup_steps, following_schedule=learning_rate)

        # B) ARCHITECTURE
        inputs = {
            "input_ids": tf.keras.layers.Input(shape=[None], dtype=tf.int32),
            "attention_mask": tf.keras.layers.Input(shape=[None], dtype=tf.int32)
        }

        eleczech_output = eleczech(inputs).last_hidden_state
        eleczech_cls_token = eleczech_output[:, 0]
        eleczech_dropout = tf.keras.layers.Dropout(args.dropout)(eleczech_cls_token)

        predictions = tf.keras.layers.Dense(3, activation=tf.nn.softmax)(eleczech_dropout)

        super().__init__(inputs=inputs, outputs=predictions)

        # C) COMPILE
        self.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                     metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")])
        # self.summary()


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

    # Load the Electra Czech small lowercased
    tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/eleczech-lc-small")
    eleczech = transformers.TFAutoModel.from_pretrained("ufal/eleczech-lc-small")

    # : Load the data. Consider providing a `tokenizer` to the
    # constructor of the TextClassificationDataset.
    facebook = TextClassificationDataset("czech_facebook")

    def create_dataset(name) -> tf.data.Dataset:
        dataset = getattr(facebook, name)
        data = dataset.data
        # text tokenization
        X = [tokenizer.encode(sentence) for sentence in data['documents']]
        max_length = max(len(sentence) for sentence in X)
        X_ids = np.zeros([len(X), max_length], dtype=np.int32)
        X_masks = np.zeros([len(X), max_length], dtype=np.int32)
        for i in range(len(X)):
            X_ids[i, :len(X[i])] = X[i]
            X_masks[i, :len(X[i])] = 1

        # labels
        if name != "test":  # only for train and dev
            y = dataset.label_mapping([label for label in data['labels']])
        else:
            y = None

        # create tf dataset
        dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': X_ids, 'attention_mask': X_masks}, y))
        if name == "train":
            dataset = dataset.shuffle(buffer_size=10000, seed=args.seed)
        dataset = dataset.batch(args.batch_size)
        return dataset

    train = create_dataset("train")
    dev = create_dataset("dev")
    test = create_dataset("test")

    # : Create the model and train it
    model = Model(args, eleczech, train)
    print(args)
    model.fit(
        train, batch_size=args.batch_size, epochs=args.epochs, validation_data=dev,
        callbacks=[tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0),
                   tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=100,
                                                    verbose=0, mode="max", baseline=None, restore_best_weights=True)]
    )
    print(args)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "sentiment_analysis.txt"), "w", encoding="utf-8") as predictions_file:
        # : Predict the tags on the test set.
        print("create test set predictions")
        predictions = model.predict(test)

        label_strings = facebook.test.label_mapping.get_vocabulary()
        for sentence in predictions:
            print(label_strings[np.argmax(sentence)], file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
