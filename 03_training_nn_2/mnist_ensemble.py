#!/usr/bin/env python3
# TEAM MEMBERS:
# Antonio Krizmanic - 2b193238-8e3c-11ec-986f-f39926f24a9c
# Jan Kilic - 079f5b80-9807-11ec-986f-f39926f24a9c
# Janek Putz - e31a3cae-8e6c-11ec-986f-f39926f24a9c

import argparse
import os
import sys
from typing import List, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layers", default=[200], nargs="*", type=int, help="Hidden layer sizes.")
parser.add_argument("--models", default=5, type=int, help="Number of models.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace) -> Tuple[List[float], List[float]]:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Load data
    mnist = MNIST()

    # Create models
    models = []
    for model in range(args.models):
        models.append(tf.keras.Sequential(
            [tf.keras.layers.Flatten(input_shape=[MNIST.H, MNIST.W, MNIST.C]),]
            + [tf.keras.layers.Dense(hidden_layer, activation=tf.nn.relu) for hidden_layer in args.hidden_layers]
            + [tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax),]
        ))

        models[-1].compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        print("Training model {}: ".format(model + 1), end="", file=sys.stderr, flush=True)
        models[-1].fit(
            mnist.train.data["images"], mnist.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs, verbose=0
        )
        print("Done", file=sys.stderr)

    individual_accuracies, ensemble_accuracies = [], []
    for model in range(args.models):
        print(f" Model {model + 1}".center(40, '*'))
        # : Compute the accuracy on the test set for
        # the individual `models[model]`.
        individual_accuracy = models[model].evaluate(mnist.test.data["images"],
                                                     mnist.test.data["labels"],
                                                     batch_size=args.batch_size,
                                                     return_dict=True)['accuracy']

        # : Compute the accuracy on the test set for
        # the ensemble `models[0:model+1].
        #
        # Generally you can choose one of the following approaches:
        # 1) Use Keras Functional API and construct a `tf.keras.Model`
        #    which averages the models in the ensemble (using for example
        #    `tf.keras.layers.Average` of manually with `tf.math.reduce_mean`).
        #    Then you can compile the model with the required metric (without
        #    an optimizer and a loss) and use `model.evaluate`.
        # 2) Manually perform the averaging (using TF or NumPy). In this case
        #    you do not need to construct Keras ensemble model at all,
        #    and instead call `model.predict` on individual models and
        #    average the results. To measure accuracy, either do it completely
        #    manually or use `tf.metrics.SparseCategoricalAccuracy`.

        # ensemble_model = Average()(models[0:model+1])
        m_preds = []
        first_preds = []
        for m in models[0:model+1]:
            m_pred = m.predict(mnist.test.data["images"])
            m_preds.append(m_pred)
            print(m_pred.shape)
            first_preds.append(m_pred[0][0])
        m_preds_average = np.average(m_preds, axis=0)
        print("avg shape", m_preds_average.shape)
        print(sum(first_preds) / len (first_preds))
        print(m_preds_average[0][0])
        accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        accuracy.update_state(mnist.test.data["labels"], m_preds_average)
        ensemble_accuracy = accuracy.result().numpy()

        # Store the accuracies
        individual_accuracies.append(individual_accuracy)
        ensemble_accuracies.append(ensemble_accuracy)
    return individual_accuracies, ensemble_accuracies

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    individual_accuracies, ensemble_accuracies = main(args)
    for model, (individual_accuracy, ensemble_accuracy) in enumerate(zip(individual_accuracies, ensemble_accuracies)):
        print("Model {}, individual accuracy {:.2f}, ensemble accuracy {:.2f}".format(
            model + 1, 100 * individual_accuracy, 100 * ensemble_accuracy))
