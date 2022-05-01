#!/usr/bin/env python3
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

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of the threads 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
# Standard params
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# Architecture params


class Model(tf.keras.Model):
    def __init__(self, args, eleczech):

        input = {
            "input_ids": tf.keras.layers.Input(shape=[None], dtype=tf.int32),
            "attention_mask": tf.keras.layers.Input(shape=[None], dtype=tf.int32)
        }

        eleczech_output = eleczech(input).last_hidden_state
        eleczech_flattened = tf.keras.layers.Flatten()(eleczech_output)

        hidden = tf.keras.layers.Dense(64, activation=tf.nn.relu)(eleczech_flattened)


        predictions = tf.keras.layers.Dense(3, activation=tf.nn.softmax)(hidden)
        print(predictions)

        super().__init__(inputs=input, outputs=predictions)

        self.compile(optimizer=tf.optimizers.Adam(),
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                     metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")])
        self.summary()


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

    def create_dataset(data: Dict[str, Any], training: bool) -> tf.data.Dataset:
        # text tokenization
        X = [tokenizer.encode(sentence) for sentence in data['documents']]
        max_length = max(len(sentence) for sentence in X)
        X_ids = np.zeros([len(X), max_length], dtype=np.int32)
        X_masks = np.zeros([len(X), max_length], dtype=np.int32)
        for i in range(len(X)):
            X_ids[i, :len(X[i])] = X[i]
            X_masks[i, :len(X[i])] = 1

        # labels
        y = np.array([label for label in data['labels']])
        def one_hot(array):
            unique, inverse = np.unique(array, return_inverse=True)
            onehot = np.eye(unique.shape[0])[inverse]
            return onehot
        y = one_hot(y)

        # create tf dataset
        dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': X_ids, 'attention_mask': X_masks}, y))
        if training:
            dataset = dataset.shuffle(buffer_size=10000, seed=args.seed)
        dataset = dataset.batch(args.batch_size)
        return dataset

    train = create_dataset(facebook.train.data, True)
    dev = create_dataset(facebook.dev.data, False)
    test = create_dataset(facebook.test.data, False)
    for b in train.take(1):
        print(b)
        break

    # exit()

    # TODO: Create the model and train it
    model = Model(args, eleczech)

    model.fit(
        train, batch_size=args.batch_size, epochs=args.epochs, validation_data=dev,
        callbacks=[tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0),
                   tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=100,
                                                    verbose=0, mode="max", baseline=None, restore_best_weights=True)]
    )
    exit()

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "sentiment_analysis.txt"), "w", encoding="utf-8") as predictions_file:
        # : Predict the tags on the test set.
        predictions = model.predict(test)

        label_strings = facebook.test.label_mapping.get_vocabulary()
        for sentence in predictions:
            print(label_strings[np.argmax(sentence)], file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
