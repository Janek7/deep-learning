#!/usr/bin/env python3
# TEAM MEMBERS:
# Antonio Krizmanic - 2b193238-8e3c-11ec-986f-f39926f24a9c
# Janek Putz - e31a3cae-8e6c-11ec-986f-f39926f24a9c
import argparse
import datetime
import os
import re
from typing import List, Tuple
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--cnn", default="5-3-2", type=str, help="CNN architecture.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--verify", default=True, action="store_true", help="Verify the implementation.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Convolution:
    def __init__(self, filters: int, kernel_size: int, stride: int, input_shape: List[int], verify: bool) -> None:
        # Create a convolutional layer with the given arguments
        # and given input shape (e.g., [28, 28, 1]).
        self._filters = filters
        self._kernel_size = kernel_size
        self._stride = stride
        self._verify = verify

        # Here the kernel and bias variables are created
        self._kernel = tf.Variable(tf.initializers.GlorotUniform(seed=42)(
            [kernel_size, kernel_size, input_shape[2], filters]))
        self._bias = tf.Variable(tf.initializers.Zeros()([filters]))

    def forward(self, inputs: tf.Tensor) -> tf.Tensor:
        # : Compute the forward propagation through the convolution
        # with `tf.nn.relu` activation, and return the result.
        #
        # In order for the computation to be reasonably fast, you cannot
        # manually iterate through the individual pixels, batch examples,
        # input filters, or output filters. However, you can manually
        # iterate through the kernel size.

        kernels = []
        for m in range(self._kernel.shape[0]):
            for n in range(self._kernel.shape[1]):
                strided_inputs = inputs[
                                 :,  # all batches
                                 m: m+inputs.shape[1]-(self._kernel.shape[0]-1): self._stride,  # width stride
                                 n: n+inputs.shape[2]-(self._kernel.shape[1]-1): self._stride,  # height stride
                                 :  # all input channels
                                 ]
                kernel_m_n = (strided_inputs @ self._kernel[m, n, :, :])
                kernels.append(kernel_m_n)

        # sum all kernels together
        output = tf.nn.relu(tf.add_n(kernels) + self._bias)

        # If requested, verify that `output` contains a correct value.
        if self._verify:
            reference = tf.nn.relu(tf.nn.convolution(inputs, self._kernel, self._stride) + self._bias)
            np.testing.assert_allclose(output, reference, atol=1e-5, err_msg="Forward pass differs!")

        return output

    def backward(
        self, inputs: tf.Tensor, outputs: tf.Tensor, outputs_gradient: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # : Given this layer's inputs, this layer's outputs,
        # and the gradient with respect to the layer's outputs,
        # compute the derivatives of the loss with respect to
        # - the `inputs` layer,
        # - `self._kernel`,
        # - `self._bias`.
        k = self._kernel.shape[0]

        # compute output gradients with relu derivation
        outputs_gradient = outputs_gradient * tf.cast(outputs > 0, tf.float32)

        # compute bias gradient
        bias_gradient = tf.reduce_sum(outputs_gradient, axis=[0, 1, 2])

        # pad for Inputs Gradient Variant A (not used)
        # pad output gradient - init with zeros and shape of input but channels of output
        # output_gradient_padded = tf.Variable(tf.zeros(inputs.shape[:3] + [outputs_gradient.shape[-1]]))
        # output_gradient_padded[:, :-(k-1):self._stride, :-(k-1):self._stride].assign(outputs_gradient)

        # compute input and kernel gradient
        kernel_gradient = tf.Variable(tf.zeros_like(self._kernel))
        # inputs_gradient_A = 0
        inputs_gradient_B = tf.Variable(tf.zeros_like(inputs))

        for m in range(k):
            for n in range(k):
                # kernel gradient
                strided_inputs = inputs[
                                 :,  # all batches
                                 m: m + inputs.shape[1] - (k - 1): self._stride,  # width stride
                                 n: n + inputs.shape[2] - (k - 1): self._stride,  # height stride
                                 :  # all input channels
                                 ]
                kernel_gradient[m, n].assign(tf.einsum("abci,abcj->ij", strided_inputs, outputs_gradient))

                # Approach A: fails because input gradient shape at the end differs
                # strided_output_gradient = output_gradient_padded[
                #                              :,  # all batches
                #                              m: m + inputs.shape[1] - (k - 1): self._stride,  # width stride
                #                              n: n + inputs.shape[2] - (k - 1): self._stride,  # height stride
                #                              :  # all input channels
                #                              ]
                # inputs_gradient_A += (strided_output_gradient @ tf.transpose(self._kernel[k - 1 - m, k - 1 - n, :, :]))

                # Approach B: do not pad output gradients before, but slice fitting inputs_gradients
                inputs_gradient_slice = inputs_gradient_B[:,
                                                           m: m + ((((inputs.shape[1] - k) // self._stride) + 1) * self._stride): self._stride,
                                                           n: n + ((((inputs.shape[2] - k) // self._stride) + 1) * self._stride): self._stride,
                                                           :]
                inputs_gradient_slice.assign(inputs_gradient_slice + outputs_gradient @ tf.transpose(self._kernel[m, n]))

        inputs_gradient = inputs_gradient_B

        # If requested, verify that the three computed gradients are correct.
        if self._verify:
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                reference = tf.nn.relu(tf.nn.convolution(inputs, self._kernel, self._stride) + self._bias)
            for name, computed, reference in zip(
                    ["Inputs", "Kernel", "Bias"], [inputs_gradient, kernel_gradient, bias_gradient],
                    tape.gradient(reference, [inputs, self._kernel, self._bias], outputs_gradient)):
                np.testing.assert_allclose(computed, reference, atol=1e-5, err_msg=name + " gradient differs!")

        # Return the inputs gradient, the layer variables, and their gradients.
        return inputs_gradient, [self._kernel, self._bias], [kernel_gradient, bias_gradient]


class Model:
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args

        # Create the convolutional layers according to `args.cnn`.
        input_shape = [MNIST.H, MNIST.W, MNIST.C]
        self._convs = []
        for layer in args.cnn.split(","):
            filters, kernel_size, stride = map(int, layer.split("-"))
            self._convs.append(Convolution(filters, kernel_size, stride, input_shape, args.verify))
            input_shape = [(input_shape[0] - kernel_size) // stride + 1,
                           (input_shape[1] - kernel_size) // stride + 1, filters]

        # Create the classification head
        self._flatten = tf.keras.layers.Flatten(input_shape=input_shape)
        self._classifier = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)

        # Create the metric and the optimizer
        self._accuracy = tf.metrics.SparseCategoricalAccuracy()
        self._optimizer = tf.optimizers.Adam(args.learning_rate)

    def train_epoch(self, dataset: MNIST.Dataset) -> None:
        for batch in dataset.batches(self._args.batch_size):
            # Forward pass through the convolutions
            hidden = tf.constant(batch["images"])
            conv_values = [hidden]
            for conv in self._convs:
                hidden = conv.forward(hidden)
                conv_values.append(hidden)

            # Run the classification head
            hidden_flat = self._flatten(hidden)
            predictions = self._classifier(hidden_flat)

            # Compute the gradients of the classifier and the convolution output
            d_logits = (predictions - tf.one_hot(batch["labels"], MNIST.LABELS)) / len(batch["images"])
            variables = [self._classifier.bias, self._classifier.kernel]
            gradients = [tf.reduce_sum(d_logits, 0), tf.tensordot(hidden_flat, d_logits, axes=[0, 0])]
            hidden_gradient = tf.reshape(tf.linalg.matvec(self._classifier.kernel, d_logits), hidden.shape)

            # Backpropagate the gradient through the convolutions
            for conv, inputs, outputs in reversed(list(zip(self._convs, conv_values[:-1], conv_values[1:]))):
                hidden_gradient, conv_variables, conv_gradients = conv.backward(inputs, outputs, hidden_gradient)
                variables.extend(conv_variables)
                gradients.extend(conv_gradients)

            # Update the weights
            self._optimizer.apply_gradients(zip(gradients, variables))

    def evaluate(self, dataset: MNIST.Dataset) -> float:
        self._accuracy.reset_states()
        for batch in dataset.batches(self._args.batch_size):
            hidden = batch["images"]
            for conv in self._convs:
                hidden = conv.forward(hidden)
            hidden = self._flatten(hidden)
            predictions = self._classifier(hidden)
            self._accuracy(batch["labels"], predictions)
        return self._accuracy.result()


def main(args: argparse.Namespace) -> float:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Load data, using only 5000 training images
    mnist = MNIST(size={"train": 5000})

    # Create the model
    model = Model(args)

    for epoch in range(args.epochs):
        model.train_epoch(mnist.train)

        accuracy = model.evaluate(mnist.dev)
        print("Dev accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * accuracy))

    accuracy = model.evaluate(mnist.test)
    print("Test accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * accuracy))

    # Return the test accuracy for ReCodEx to validate.
    return accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)