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
import tensorflow_addons as tfa

from morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
parser.add_argument("--rnn_cell_dim", default=64, type=int, help="RNN cell dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=128, type=int, help="Word embedding dimension.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        # Implement a one-layer RNN network. The input `words` is
        # a RaggedTensor of strings, each batch example being a list of words.
        words = tf.keras.layers.Input(shape=[None], dtype=tf.string, ragged=True)

        # (tagger_crf): Map strings in `words` to indices by using the `word_mapping` of `train.forms`.
        words_indices = train.forms.word_mapping(words)

        # (tagger_crf): Embed input words with dimensionality `args.we_dim`. Note that the `word_mapping`
        # provides a `vocabulary_size()` call returning the number of unique words in the mapping.
        embeddings = tf.keras.layers.Embedding(train.forms.word_mapping.vocabulary_size(), args.we_dim)(words_indices)

        # (tagger_crf): Create the specified `args.rnn_cell` RNN cell (LSTM, GRU) with
        # dimension `args.rnn_cell_dim`. The cell should produce an output for every
        # sequence element (so a 3D output). Then apply it in a bidirectional way on
        # the embedded words, **summing** the outputs of forward and backward RNNs.
        if args.rnn_cell == 'LSTM':
            cell_type = tf.keras.layers.LSTM
        elif args.rnn_cell == 'GRU':
            cell_type = tf.keras.layers.GRU
        else:
            raise NotImplementedError(f"{args.rnn_cell} is not a valid RNN cell")
        rnn_forward_seq = cell_type(units=args.rnn_cell_dim, return_sequences=True)(embeddings)
        rnn_backward_seq = cell_type(units=args.rnn_cell_dim, return_sequences=True, go_backwards=True)(embeddings)
        rnn_backward_seq = tf.reverse(rnn_backward_seq, axis=[1])
        rnn_sequences = tf.keras.layers.Add()([rnn_forward_seq, rnn_backward_seq])

        # (tagger_crf): Add a final classification layer into as many classes as there are unique
        # tags in the `word_mapping` of `train.tags`. Note that **no activation** should
        # be used, the CRF operations will take care of it.
        predictions = tf.keras.layers.Dense(train.tags.word_mapping.vocabulary_size())(rnn_sequences)

        # Check that the created predictions are a 3D tensor.
        assert predictions.shape.rank == 3
        super().__init__(inputs=words, outputs=predictions)

        # We compile the model with CRF loss and SpanLabelingF1 metric.
        self.compile(optimizer=tf.optimizers.Adam(),
                     loss=self.crf_loss,
                     metrics=[self.SpanLabelingF1Metric(train.tags.word_mapping.get_vocabulary(), name="f1")])

        # TODO(tagger_crf): Create `self._crf_weights`, a trainable zero-initialized tf.float32 matrix variable
        # of size [number of unique train tags, number of unique train.tags], using `self.add_weight`.
        self._crf_weights = self.add_weight("crf_weights", shape=[train.tags.word_mapping.vocabulary_size(),
                                                                  train.tags.word_mapping.vocabulary_size()],
                                            initializer=tf.initializers.Zeros(), dtype=tf.float32)

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    def crf_loss(self, gold_labels: tf.RaggedTensor, logits: tf.RaggedTensor) -> tf.Tensor:
        assert isinstance(gold_labels, tf.RaggedTensor), "Gold labels given to CRF loss must be RaggedTensors"
        assert isinstance(logits, tf.RaggedTensor), "Logits given to CRF loss must be RaggedTensors"

        # TODO: Implement the CRF loss computation manually, without using the `tfa.text` methods.
        # You can count on the fact that all training sentences contain at least 2 words.
        #
        # The following remarks might come handy:
        # - Custom RNN cells can be implemented by deriving from tf.keras.layers.AbstractRNNCell
        #   and definining at least `state_size`, `output_size` and `call`:
        #
        #     class CRFCell(tf.keras.layers.AbstractRNNCell):
        #         @property
        #         def output_size(self):
        #             # Return output dimensionality as a scalar number
        #         @property
        #         def state_size(self):
        #             # Return state dimensionality as either a scalar number or a vector
        #         def call(self, inputs, states):
        #             # Given inputs from the current timestep and states from the previous one,
        #             # return an (outputs, new_states) pair. Note that `states` and `new_states`
        #             # must always be a tuple of tensors, even if there is only a single state.
        #
        #   Such a cell can then be used by the `tf.keras.layers.RNN` layer. If you want to
        #   specify a different initial state then all zeros, pass it as `initial_state` argument
        #   along with the inputs.
        #
        # - RaggedTensors cannot be directly indexed in the ragged dimension, but they can be sliced.
        #   For example, to skip the first word in gold_labels, you can call
        #     gold_labels[:, 1:]
        #   but to get the first word in gold_labels, you cannot use
        #     gold_labels[:, 0]
        #   If you really require indexing in the ragged dimension, convert them to dense tensors.
        #
        # - To index a (possible ragged) tensor with another (possible ragged) tensor,
        #   `tf.gather` and `tf.gather_nd` can be used. If is useful fo pay attention
        #   to the `batch_dims` argument of these calls.
        raise NotImplementedError()

    def crf_decode(self, logits: tf.RaggedTensor) -> tf.RaggedTensor:
        assert isinstance(logits, tf.RaggedTensor), "Logits given to CTC decoding must be RaggedTensors"

        # (tagger_crf): Perform CRF decoding using `tfa.text.crf_decode`. Convert the
        # logits analogously as in `crf_loss`. Finally, convert the result
        # to a ragged tensor.
        decode_tags, _ = tfa.text.crf_decode(potentials=logits.to_tensor(), transition_params=self._crf_weights,
                                             sequence_length=logits.row_lengths())
        predictions = tf.RaggedTensor.from_tensor(decode_tags, lengths=logits.row_lengths())

        assert isinstance(predictions, tf.RaggedTensor)
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

    # We override `predict_step` to run CRF decoding during prediction.
    def predict_step(self, data):
        data = data[0] if isinstance(data, tuple) else data
        y_pred = self(data, training=False)
        y_pred = self.crf_decode(y_pred)
        return y_pred

    # We override `test_step` to run CRF decoding during evaluation.
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compute_loss(x, y, y_pred)
        y_pred = self.crf_decode(y_pred)
        return self.compute_metrics(x, y, y_pred, None)

    class SpanLabelingF1Metric(tf.metrics.Metric):
        """Keras-like metric evaluating span labeling F1-score of RaggedTensors."""
        def __init__(self, tags, name="span_labeling_f1", dtype=None):
            super().__init__(name, dtype)
            self._tags = tags
            self._counts = self.add_weight("counts", shape=[3], initializer=tf.initializers.Zeros(), dtype=tf.int64)

        def reset_state(self):
            self._counts.assign([0] * 3)

        def classify_spans(self, y_true, y_pred, sentence_limits):
            sentence_limits = set(sentence_limits)
            spans_true, spans_pred = set(), set()
            for spans, labels in [(spans_true, y_true), (spans_pred, y_pred)]:
                span = None
                for i, label in enumerate(self._tags[label] for label in labels):
                    if span and (label.startswith(("O", "B")) or i in sentence_limits):
                        spans.add((start, i, span))
                        span = None
                    if label.startswith("B"):
                        span, start = label[2:], i
                if span:
                    spans.add((start, len(labels), span))
            return np.array([len(spans_true & spans_pred), len(spans_pred - spans_true),
                             len(spans_true - spans_pred)], np.int64)

        def update_state(self, y, y_pred, sample_weight=None):
            assert isinstance(y, tf.RaggedTensor) and isinstance(y_pred, tf.RaggedTensor)
            assert sample_weight is None, "sample_weight currently not supported"
            counts = tf.numpy_function(self.classify_spans, (y.values, y_pred.values, y.row_limits()), tf.int64)
            self._counts.assign_add(counts)

        def result(self):
            tp, fp, fn = self._counts[0], self._counts[1], self._counts[2]
            return tf.math.divide_no_nan(tf.cast(2 * tp, tf.float32), tf.cast(2 * tp + fp + fn, tf.float32))


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
    morpho = MorphoDataset("czech_cnec", max_sentences=args.max_sentences)

    # Create the model and train
    model = Model(args, morpho.train)

    # (tagger_crf): Construct dataset for training, which should contain pairs of
    # - tensor of string words (forms) as input
    # - tensor of integral tag ids as targets.
    # To create the identifiers, use the `word_mapping` of `morpho.train.tags`.
    def tagging_dataset(example):
        return example['forms'], morpho.train.tags.word_mapping(example['tags'])

    def create_dataset(name):
        dataset = getattr(morpho, name).dataset
        dataset = dataset.map(tagging_dataset)
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        return dataset
    train, dev = create_dataset("train"), create_dataset("dev")

    logs = model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[model.tb_callback])

    # Return all metrics for ReCodEx to validate
    return {metric: values[-1] for metric, values in logs.history.items()}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
