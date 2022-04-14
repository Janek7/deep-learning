#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from typing import Dict
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
# TODO
tf.config.run_functions_eagerly(True)
import tensorflow_addons as tfa

from morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=2, type=int, help="Batch size.")
parser.add_argument("--cle_dim", default=32, type=int, help="CLE embedding dimension.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=500, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn_dim", default=32, type=int, help="RNN cell dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()

        self.source_mapping = train.forms.char_mapping
        self.target_mapping = train.lemmas.char_mapping
        self.target_mapping_inverse = type(self.target_mapping)(
            vocabulary=self.target_mapping.get_vocabulary(), invert=True)

        # : Define
        # - `self.source_embedding` as an embedding layer of source chars into `args.cle_dim` dimensions
        # - `self.source_rnn` as a bidirectional GRU with `args.rnn_dim` units, returning only the last output,
        #   summing opposite directions
        self.source_embedding = tf.keras.layers.Embedding(self.source_mapping.vocabulary_size(), args.cle_dim)
        self.source_rnn = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(args.rnn_dim), merge_mode='sum')

        # : Then define
        # - `self.target_embedding` as an embedding layer of target chars into `args.cle_dim` dimensions
        # - `self.target_rnn_cell` as a GRUCell with `args.rnn_dim` units
        # - `self.target_output_layer` as a Dense layer into as many outputs as there are unique target chars
        self.target_embedding = tf.keras.layers.Embedding(self.target_mapping.vocabulary_size(), args.cle_dim)
        self.target_rnn_cell = tf.keras.layers.GRUCell(args.rnn_dim)
        self.target_output_layer = tf.keras.layers.Dense(self.target_mapping.vocabulary_size())

        # Compile the model
        self.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.Accuracy(name="accuracy")],
        )

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    class DecoderTraining(tfa.seq2seq.BaseDecoder):
        """
        returns predicted indices at inference
        """
        def __init__(self, lemmatizer, *args, **kwargs):
            self.lemmatizer = lemmatizer
            super().__init__.__wrapped__(self, *args, **kwargs)

        @property
        def batch_size(self):
            # : Return the batch size of `self.source_states` as a *scalar* number;
            # use `tf.shape` to get the full shape and then extract the batch size.
            return tf.shape(self.source_states)[0]

        @property
        def output_size(self):
            # : Describe the size of a single decoder output (batch size and the
            # sequence length are not included) by returning
            #   tf.TensorShape(number of logits of each output element [lemma character])
            return tf.TensorShape([self.lemmatizer.target_mapping.vocabulary_size()])

        @property
        def output_dtype(self):
            # : Return the type of the decoder output (so the type of the produced logits).
            return tf.float32

        def initialize(self, layer_inputs, initial_state=None):
            self.source_states, self.targets = layer_inputs

            # : Define `finished` as a vector of self.batch_size of `False` [see tf.fill].
            finished = tf.fill([self.batch_size], False)

            # : Define `inputs` as a vector of self.batch_size of MorphoDataset.Factor.BOW,
            # embedded using self.lemmatizer.target_embedding
            inputs = self.lemmatizer.target_embedding(tf.fill([self.batch_size], MorphoDataset.Factor.BOW))

            # : Define `states` as self.source_states
            states = self.source_states

            return finished, inputs, states

        def step(self, time, inputs, states, training):
            # : Pass `inputs` and `[states]` through self.lemmatizer.target_rnn_cell,
            # which returns `(outputs, [states])`.
            outputs, [states] = self.lemmatizer.target_rnn_cell(inputs, [states])

            # : Overwrite `outputs` by passing them through self.lemmatizer.target_output_layer,
            outputs = self.lemmatizer.target_output_layer(outputs)

            # : Define `next_inputs` by embedding `time`-th chars from `self.targets`.
            # [:, time] -> : is full batch dim; only time-th letter -> this includes all before because of transitive
            # dependencies of the predictions
            next_inputs = self.lemmatizer.target_embedding(self.targets[:, time])

            # : Define `finished` as a vector of booleans; True if the corresponding
            # `time`-th char from `self.targets` is `MorphoDataset.Factor.EOW`, False otherwise.
            finished = tf.math.equal(self.targets[:, time], MorphoDataset.Factor.EOW)

            return outputs, states, next_inputs, finished

    class DecoderPrediction(DecoderTraining):
        """
        returns predicted indices at inference
        """
        @property
        def output_size(self):
            # : Describe the size of a single decoder output (batch size and the
            # sequence length are not included) by returning a suitable
            # `tf.TensorShape` representing a *scalar* element, because we are producing
            # lemma character indices during prediction.
            return tf.TensorShape([])

        @property
        def output_dtype(self):
            # : Return the type of the decoder output (i.e., target lemma character indices).
            return tf.int32

        def initialize(self, layer_inputs, initial_state=None):
            # Use `initialize` from the DecoderTraining, passing None as targets
            return super().initialize([layer_inputs, None], initial_state)

        def step(self, time, inputs, states, training):
            # (DecoderTraining): Pass `inputs` and `[states]` through self.lemmatizer.target_rnn_cell,
            # which returns `(outputs, [states])`.
            outputs, [states] = self.lemmatizer.target_rnn_cell(inputs, [states])

            # : Overwrite `outputs` by passing them through self.lemmatizer.target_output_layer,
            outputs = self.lemmatizer.target_output_layer(outputs)

            # : Overwrite `outputs` by passing them through `tf.argmax` on suitable axis and with
            # `output_type=tf.int32` parameter.
            outputs = tf.math.argmax(outputs, axis=1, output_type=tf.int32)

            # : Define `next_inputs` by embedding the `outputs`
            next_inputs = self.lemmatizer.target_embedding(outputs)

            # : Define `finished` as a vector of booleans; True if the corresponding
            # prediction in `outputs` is `MorphoDataset.Factor.EOW`, False otherwise.
            finished = tf.math.equal(outputs, MorphoDataset.Factor.EOW)

            return outputs, states, next_inputs, finished

    # If `targets` is given, we are in the teacher forcing mode.
    # Otherwise, we run in autoregressive mode.
    def call(self, inputs, targets=None):
        # Forget about sentence boundaries and instead consider
        # all valid form-lemma pairs as independent batch examples.
        #
        # Then, split the given forms into character sequences and map then
        # to their indices.
        source_charseqs = inputs.values
        source_charseqs = tf.strings.unicode_split(source_charseqs, "UTF-8")
        source_charseqs = self.source_mapping(source_charseqs)
        if targets is not None:
            # The targets are already mapped sequences of characters, so only
            # drop the sentence boundaries, and convert to a dense tensor
            # (the EOW correctly indicate end of lemma).
            target_charseqs = targets.values
            target_charseqs = target_charseqs.to_tensor()

        # : Embed source_charseqs using `source_embedding`
        source_charseqs_embeddings = self.source_embedding(source_charseqs)

        # : Run source_rnn on the embedded sequences, returning outputs in `source_states`.
        sources_states = self.source_rnn(source_charseqs_embeddings)

        # Run the appropriate decoder. Note that the outputs of the decoders
        # are exactly the outputs of `tfa.seq2seq.dynamic_decode`.
        if targets is not None:
            # : Create a self.DecoderTraining by passing `self` to its constructor.
            # Then run it on `[source_states, target_charseqs]` input,
            # storing the first result in `output` and the third result in `output_lens`.
            decoder_training = self.DecoderTraining(self)
            result = decoder_training([sources_states, target_charseqs])
            output, output_lens = result[0], result[2]
        else:
            # : Create a self.DecoderPrediction by using:
            # - `self` as first argument to its constructor
            # - `maximum_iterations=tf.cast(source_charseqs.bounding_shape(1) + 10, tf.int32)`
            #   as another argument, which indicates that the longest prediction
            #   must be at most 10 characters longer than the longest input.
            #
            # Then run it on `source_states`, storing the first result in `output`
            # and the third result in `output_lens`. Finally, because we do not want
            # to return the `[EOW]` symbols, decrease `output_lens` by one.
            decoder_prediction = self.DecoderPrediction(self,  maximum_iterations=tf.cast(
                source_charseqs.bounding_shape(1) + 10, tf.int32))
            result = decoder_prediction(sources_states)
            output, output_lens = result[0], result[2]
            output_lens -= 1

        # Reshape the output to the original matrix of lemmas
        # and explicitly set mask for loss and metric computation.
        output = tf.RaggedTensor.from_tensor(output, output_lens)
        output = inputs.with_values(output)
        return output

    def train_step(self, data):
        x, y = data

        # Convert `y` by splitting characters, mapping characters to ids using
        # `self.target_mapping` and finally appending `MorphoDataset.Factor.EOW`
        # to every sequence.
        y_targets = self.target_mapping(tf.strings.unicode_split(y.values, "UTF-8"))
        y_targets = tf.concat(
            [y_targets, tf.fill([y_targets.bounding_shape(0), 1],
                                tf.constant(MorphoDataset.Factor.EOW, tf.int64))], axis=-1)
        y_targets = y.with_values(y_targets)

        with tf.GradientTape() as tape:
            y_pred = self(x, targets=y_targets, training=True)
            loss = self.compute_loss(x, y_targets.values, y_pred.values)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return {"loss": metric.result() for metric in self.metrics if metric.name == "loss"}

    def predict_step(self, data):
        if isinstance(data, tuple): data = data[0]
        y_pred = self(data, training=False)
        y_pred = self.target_mapping_inverse(y_pred)
        y_pred = tf.strings.reduce_join(y_pred, axis=-1)
        return y_pred

    def test_step(self, data):
        x, y = data
        y_pred = self.predict_step(data)
        self.compiled_metrics.update_state(
            tf.ones_like(y.values, dtype=tf.int32), tf.cast(y_pred.values == y.values, tf.int32))
        return {m.name: m.result() for m in self.metrics if m.name != "loss"}


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
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences, add_bow_eow=True)

    # Create the model and train
    model = Model(args, morpho.train)

    # Construct dataset for lemmatizer training
    def create_dataset(name):
        dataset = getattr(morpho, name).dataset
        dataset = dataset.map(lambda example: (example["forms"], example["lemmas"]))
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        return dataset
    train, dev = create_dataset("train"), create_dataset("dev")

    # Callback showing intermediate results during training
    class ShowIntermediateResults(tf.keras.callbacks.Callback):
        def __init__(self, data):
            self._iterator = iter(data.repeat())
        def on_train_batch_end(self, batch, logs=None):
            if model.optimizer.iterations % 10 == 0:
                forms, lemmas = next(self._iterator)
                print(model.optimizer.iterations.numpy(),
                      *[repr(strings[0, 0].numpy().decode("utf-8"))
                        for strings in [forms, lemmas, model.predict_on_batch(forms[:1, :1])]])

    logs = model.fit(train, epochs=args.epochs, validation_data=dev, verbose=2,
                     callbacks=[ShowIntermediateResults(dev), model.tb_callback])

    # Return all metrics for ReCodEx to validate
    return {metric: values[-1] for metric, values in logs.history.items()}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
