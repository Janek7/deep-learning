#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from typing import Dict
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=800, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--transformer_dropout", default=0., type=float, help="Transformer dropout.")
parser.add_argument("--transformer_expansion", default=4, type=float, help="Transformer FFN expansion factor.")
parser.add_argument("--transformer_heads", default=4, type=int, help="Transformer heads.")
parser.add_argument("--transformer_layers", default=0, type=int, help="Transformer layers.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(tf.keras.Model):
    class FFN(tf.keras.layers.Layer):
        def __init__(self, dim, expansion, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dim, self.expansion = dim, expansion
            # : Create the required layers -- first a ReLU-activated dense
            # layer with `dim * expansion` units, followed by a dense layer
            # with `dim` units without an activation.
            self._first_dense = tf.keras.layers.Dense(dim * expansion,activation=tf.nn.relu)
            self._second_dense = tf.keras.layers.Dense(dim, activation=None)

        def get_config(self):
            return {"dim": self.dim, "expansion": self.expansion}

        def call(self, inputs):
            # : Execute the FFN Transformer layer.
            return self._second_dense(self._first_dense(inputs))

    class SelfAttention(tf.keras.layers.Layer):
        def __init__(self, dim, heads, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dim, self.heads = dim, heads
            # : Create weight matrices W_Q, W_K, W_V and W_O using `self.add_weight`,
            # each with shape `[dim, dim]`; for other arguments, keep the default values
            # (which mean trainable float32 matrices initialized with `"glorot_uniform"`).
            for weight_matrix in ["W_Q", "W_K", "W_V", "W_O"]:
                setattr(self, weight_matrix, self.add_weight("crf_weights", shape=[dim, dim], trainable=True,
                                                             initializer='glorot_uniform', dtype=tf.float32))

        def get_config(self):
            return {"dim": self.dim, "heads": self.heads}

        def call(self, inputs, mask):
            mask_shape = tf.shape(mask)
            batch_size, max_sentence_len = mask_shape[0], mask_shape[1]

            # : Execute the self-attention layer.
            #
            # Start by computing Q, K and V. In all cases:
            # - first multiply `inputs` by the corresponding weight matrix W_Q/W_K/W_V,
            # - reshape via `tf.reshape` to [batch_size, max_sentence_len, heads, dim // heads],
            # - transpose via `tf.transpose` to [batch_size, heads, max_sentence_len, dim // heads].
            def compute(X, weight_matrix):
                mul = X @ weight_matrix
                reshaped = tf.reshape(mul, [batch_size, max_sentence_len, self.heads, self.dim // self.heads])
                transposed = tf.transpose(reshaped, [0, 2, 1, 3])
                return transposed

            Q = compute(inputs, self.W_Q)
            K = compute(inputs, self.W_K)
            V = compute(inputs, self.W_V)

            # : Continue by computing the self-attention weights as Q @ K^T,
            # normalizing by the square root of `dim // heads`.
            self_attention_weights = (Q @ tf.transpose(K)) / (tf.math.sqrt(self.dim // self.heads))
            self_attention_weights = self_attention_weights[:, :, tf.newaxis, :]

            # : Apply the softmax, but including a suitable mask, which ignores all padding words.
            # The original `mask` is a bool matrix of shape [batch_size, max_sentence_len]
            # indicating which words are valid (True) or padding (False).
            # - You can perform the masking manually, by setting the attention weights
            #   of padding words to -1e9.
            # - Alternatively, you can use the fact that tf.keras.layers.Softmax accepts a named
            #   boolean argument `mask` indicating the valid (True) or padding (False) elements.
            softmaxed = tf.keras.layers.Softmax()(self_attention_weights, mask=mask)

            # : Finally,
            # - take a weighted combination of values V according to the computed attention
            #   (using a suitable matrix multiplication),
            # - transpose the result to [batch_size, max_sentence_len, heads, dim // heads],
            # - reshape to [batch_size, max_sentence_len, dim],
            # - multiply the result by the W_O matrix.
            values_mul = softmaxed @ V
            values_mul_transposed = tf.transpose(values_mul, perm=[0, 2, 1, 3])
            values_mul_reshaped = tf.reshape(values_mul_transposed, [batch_size, max_sentence_len, self.dim])
            return values_mul_reshaped @ self.W_O

    class Transformer(tf.keras.layers.Layer):
        def __init__(self, layers, dim, expansion, heads, dropout, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.layers, self.dim, self.expansion, self.heads, self.dropout = layers, dim, expansion, heads, dropout
            # : Create the required number of transformer layers, each consisting of
            # - a layer normalization and a self-attention layer followed by a dropout layer,
            # - a layer normalization and a FFN layer followed by a dropout layer.
            # Note: create attention and ffn layer here central, because weights have to be stored central in the layer
            # and not be created during every call. LayerNorm and Dropout do not have trainable parameters.
            self.transformer_layers = [(Model.SelfAttention(dim, heads), Model.FNN(dim, expansion))
                                       for i in range(layers)]

        def get_config(self):
            return {name: getattr(self, name) for name in ["layers", "dim", "expansion", "heads", "dropout"]}

        def call(self, inputs, mask):
            # TODO: Start by computing the sinusoidal positional embeddings.
            # They have a shape `[max_sentence_len, dim]` and
            # - for `i < dim / 2`, the value on index `[pos, i]` should be
            #     `sin(pos / 10000 ** (2 * i / dim))`
            # - the value on index `[pos, i]` for `i >= dim / 2` should be
            #     `cos(pos / 10000 ** (2 * (i - dim/2) / dim))`
            # This order is the same as in the visualization on the slides, but
            # different from the original paper where `sin` and `cos` interleave.

            def create_positional_embeddings(batch_size, max_sentence_length):
                boundary = tf.cast(self.dim / 2, tf.int32)

                # create index matrix
                def create_idx_matrix(lower_boundary, upper_boundary):
                    max_sentence_length_indexes = tf.range(lower_boundary, upper_boundary, dtype=tf.float32)
                    return tf.transpose(tf.reshape(tf.repeat(max_sentence_length_indexes,
                                                             repeats=max_sentence_length),
                                                   shape=[boundary, max_sentence_length]),
                                        perm=[1, 0])

                i_smaller = create_idx_matrix(0, boundary)
                i_greater = create_idx_matrix(boundary, self.dim)

                # create position matrix
                pos = tf.reshape(tf.repeat(tf.range(max_sentence_length, dtype=tf.float32), boundary),
                                 shape=[max_sentence_length, boundary])

                # compute positional embeddings
                part_1 = tf.math.sin(pos / (10000 ** (2 * i_smaller / self.dim)))
                part_2 = tf.math.cos(pos / (10000 ** (2 * i_greater / self.dim)))
                positional_embeddings = tf.concat([part_1, part_2], axis=1)

                # expand to batch size (repeat batch_size times on axis 0 and 1 on axis 1)
                positional_embeddings_batch = tf.reshape(tf.tile(positional_embeddings, multiples=[batch_size, 1]),
                                                         shape=[batch_size, max_sentence_length, self.dim])
                return positional_embeddings_batch

            # inputs shape: batch size, tokens(var), dim
            batch_size, max_sentence_len = tf.shape(inputs)[0], tf.shape(inputs)[1]
            positional_embeddings = create_positional_embeddings(batch_size, max_sentence_len)

            # : Add the positional embeddings to the `inputs` and then
            # perform the given number of transformer layer, composed of
            # - a self-attention sub-layer, followed by
            # - a FFN sub-layer.
            # In each sub-layer, pass the input through LayerNorm, then compute
            # the corresponding operation, apply dropout, and finally add this result
            # to the original sub-layer input. Note that the given `mask` should be
            # passed to the self-attention operation to ignore the padding words.
            inputs += positional_embeddings

            # method for calling one transformer (encoding) layer
            def call_transformer_layer(inputs, self_attention_layer, ffn_layer):
                # attention sublayer
                layer_norm1 = tf.keras.layers.LayerNormalization()(inputs)
                self_attention = self_attention_layer(layer_norm1, mask=mask)
                dropout1 = tf.keras.layers.Dropout(self.dropout)(self_attention)
                intermediate = inputs + dropout1

                # FFN sublayer
                layer_norm2 = tf.keras.layers.LayerNormalization()(intermediate)
                ffn = ffn_layer(layer_norm2)
                dropout2 = tf.keras.layers.Dropout(self.dropout)(ffn)
                result = intermediate + dropout2

                return result

            layer_input = inputs
            for i in range(self.layers):
                self_attention_layer, ffn_layer = self.transformer_layers[i]
                layer_input = call_transformer_layer(layer_input, self_attention_layer, ffn_layer)

            return layer_input

    def __init__(self, args, train):
        # Implement a transformer encoder network. The input `words` is
        # a RaggedTensor of strings, each batch example being a list of words.
        words = tf.keras.layers.Input(shape=[None], dtype=tf.string, ragged=True)

        # (tagger_we): Map strings in `words` to indices by using the `word_mapping` of `train.forms`.
        words_indices = train.forms.word_mapping(words)

        # (tagger_we): Embed input words with dimensionality `args.we_dim`. Note that the `word_mapping`
        # provides a `vocabulary_size()` call returning the number of unique words in the mapping.
        embeddings = tf.keras.layers.Embedding(train.forms.word_mapping.vocabulary_size(), args.we_dim)(words_indices)

        # : Call the Transformer layer:
        # - create a `Model.Transformer` layer, using suitable options from `args`
        #   (using `args.we_dim` for the `dim` argument),
        # - when calling the layer, convert the ragged tensor with the input words embedding
        #   to a dense one, and also pass the following argument as a mask:
        #     `mask=tf.sequence_mask(ragged_tensor_with_input_words_embeddings.row_lengths())`
        # - finally, convert the result back to a ragged tensor.
        transformer = Model.Transformer(layers=args.transformer_layers,
                                        dim=args.we_dim,
                                        expansion=args.transformer_expansion,
                                        heads=args.transformer_heads,
                                        dropout=args.transformer_dropout)
        transformer_output = transformer(embeddings.to_tensor(),
                                         mask=tf.sequence_mask(embeddings.row_lengths()))
        transformer_output_ragged = tf.RaggedTensor.from_tensor(transformer_output, lengths=embeddings.row_lengths())

        # (tagger_we): Add a softmax classification layer into as many classes as there are unique
        # tags in the `word_mapping` of `train.tags`. Note that the Dense layer can process
        # a `RaggedTensor` without any problem.
        predictions = tf.keras.layers.Dense(train.tags.word_mapping.vocabulary_size(),
                                            activation=tf.nn.softmax)(transformer_output_ragged)

        # Check that the created predictions are a 3D tensor.
        assert predictions.shape.rank == 3
        super().__init__(inputs=words, outputs=predictions)
        self.compile(optimizer=tf.optimizers.Adam(),
                     loss=lambda yt, yp: tf.losses.SparseCategoricalCrossentropy()(yt.values, yp.values),
                     metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")])

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)


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
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences)

    # Create the model and train
    model = Model(args, morpho.train)

    # (tagger_we): Construct the data for the model, each consisting of the following pair:
    # - a tensor of string words (forms) as input,
    # - a tensor of integral tag ids as targets.
    # To create the tag ids, use the `word_mapping` of `morpho.train.tags`.
    def extract_tagging_data(example):
        return example['forms'], morpho.train.tags.word_mapping(example['tags'])

    def create_dataset(name):
        dataset = getattr(morpho, name).dataset
        dataset = dataset.map(extract_tagging_data)
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        return dataset
    train, dev = create_dataset("train"), create_dataset("dev")

    logs = model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[model.tb_callback])

    # Return development and training losses for ReCodEx to validate
    return {metric: values[-1] for metric, values in logs.history.items() if "loss" in metric}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
