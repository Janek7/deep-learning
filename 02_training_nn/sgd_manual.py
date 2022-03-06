
# TEAM MEMBERS:
# Antonio Krizmanic - 2b193238-8e3c-11ec-986f-f39926f24a9c
# Janek Putz - e31a3cae-8e6c-11ec-986f-f39926f24a9c

#PS C:\Users\Antonio\Documents\npfl114\labs\02> set-ExecutionPolicy -ExecutionPolicy remoteSigned -Scope Process
#PS C:\Users\Antonio\Documents\npfl114\labs\02> .second\scripts\activate

import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=2, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer", default=20, type=int, help="Size of the hidden layer.")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.

class Model(tf.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args

        self._W1 = tf.Variable(tf.random.normal([MNIST.W * MNIST.H * MNIST.C, args.hidden_layer], stddev=0.1, seed=args.seed), trainable=True)
        self._b1 = tf.Variable(tf.zeros([args.hidden_layer]), trainable=True)
        self._W2 = tf.Variable(tf.random.normal([args.hidden_layer, MNIST.LABELS], stddev=0.1, seed=args.seed),
                               trainable=True)
        self._b2 = tf.Variable(tf.zeros([MNIST.LABELS]), trainable=True)

    def predict(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = tf.reshape(inputs, shape=[inputs.shape[0], -1])
        hidden = inputs @ self._W1 + self._b1
        hidden = tf.nn.tanh(hidden)
        hidden2 = hidden @ self._W2 + self._b2
        output = tf.nn.softmax(hidden2)
        return inputs, hidden, output

    def not_scarse(self, arr: np.array) -> tf.Tensor:
        tmp = np.zeros(10, dtype = np.float32)
        tmp[arr[0]] = 1
        a = tf.convert_to_tensor([tmp])

        for i in range(1,len(arr)):
            tmp = np.zeros(10, dtype = np.float32)
            tmp[arr[i]] = 1
            a = tf.concat([a, [tmp]], 0)
        return a

    def train_epoch(self, dataset: MNIST.Dataset) -> None:
        for i, batch in enumerate(dataset.batches(self._args.batch_size)):
            input, hidden, output = self.predict(batch["images"])
            derivative_loss_yin = tf.math.subtract(output, self.not_scarse(batch["labels"]))
            derivative_loss_yin_averaged = tf.reduce_mean(derivative_loss_yin, axis=0)
            derivative_loss_w2 = tf.einsum("ai,aj->aij", hidden, derivative_loss_yin)
            gradients_w2 = tf.reduce_mean(derivative_loss_w2, axis=0)
            
            derivative_loss_b1 = tf.math.multiply(tf.linalg.matmul(derivative_loss_yin, tf.transpose(self._W2)), np.ones(hidden.shape) - hidden**2)

            gradients_b1 = tf.reduce_mean(derivative_loss_b1, axis = 0)
            derivative_loss_w1 = tf.einsum("ai,aj->aij", input, derivative_loss_b1)
            gradients_w1 = tf.reduce_mean(derivative_loss_w1, axis = 0)
            self._W2.assign_sub(self._args.learning_rate * gradients_w2)
            self._b2.assign_sub(self._args.learning_rate * derivative_loss_yin_averaged)
            self._b1.assign_sub(self._args.learning_rate * gradients_b1)
            self._W1.assign_sub(self._args.learning_rate * gradients_w1)

            # self._W1.assign_sub(self._args.learning_rate * gradients_w1)

    def evaluate(self, dataset: MNIST.Dataset) -> float:
        # Compute the accuracy of the model prediction
        correct = 0
        for batch in dataset.batches(self._args.batch_size):
            # : Compute the probabilities of the batch images
            input, hidden, probabilities = self.predict(batch["images"])

            # : Evaluate how many batch examples were predicted
            # correctly and increase `correct` variable accordingly.
            for i in (tf.math.argmax(probabilities, 1).numpy() == batch["labels"]):
                if i:
                    correct += 1

        return correct / dataset.size


def main(args: argparse.Namespace) -> float:
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

    # Create the TensorBoard writer
    writer = tf.summary.create_file_writer(args.logdir, flush_millis=10*1000)

    # Create the model
    model = Model(args)

    for epoch in range(args.epochs):
        # (sgd_backpropagation): Run the `train_epoch` with `mnist.train` dataset
        model.train_epoch(mnist.train)

        # (sgd_backpropagation): Evaluate the dev data using `evaluate` on `mnist.dev` dataset
        accuracy = model.evaluate(mnist.dev)
        print("Dev accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * accuracy), flush=True)
        with writer.as_default(step=epoch + 1):
            tf.summary.scalar("dev/accuracy", 100 * accuracy)

    # (sgd_backpropagation): Evaluate the test data using `evaluate` on `mnist.test` dataset
    accuracy = model.evaluate(mnist.test)
    print("Test accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * accuracy), flush=True)
    with writer.as_default(step=epoch + 1):
        tf.summary.scalar("test/accuracy", 100 * accuracy)

    # Return test accuracy for ReCodEx to validate
    return accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
