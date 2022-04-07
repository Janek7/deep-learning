#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
import fasttext.util
import fasttext

from morpho_analyzer import MorphoAnalyzer
from morpho_dataset import MorphoDataset

# : Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
# A standard params
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
# B tagger_cle params
parser.add_argument("--cle_dim", default=16, type=int, help="CLE embedding dimension.")
parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
parser.add_argument("--rnn_cell_dim", default=16, type=int, help="RNN cell dimension.")
parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")
parser.add_argument("--word_masking", default=0.1, type=float, help="Mask words with the given probability.")
parser.add_argument("--rnn_layers", default=1, type=int, help="Number of bidirectional rnn layers behind each other.")
parser.add_argument("--rnn_residual", default=None, type=str, help="where to create a residual connection.")
# C more params
parser.add_argument("--l2", default=None, type=float, help="L2 regularization.")
parser.add_argument("--decay", default="None", type=str, help="Learning decay rate type")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default=0.0001, type=float, help="Final learning rate.")
# allowed values for 'fasttext_mode':
#   concat (concat to existing word embeddings)
#   replace (replace existing word embeddings)
#   None (not use it at all)
parser.add_argument("--fasttext_mode", default=None, type=str, help="How to use fast text.")
parser.add_argument("--fasttext_dim", default=100, type=int, help="Dimension of fasttext vectors.")


# A layer retrieving fast text vectors for word.
class FastText(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim
        # load check model
        fasttext.util.download_model("cs", if_exists="ignore")
        self.ft = fasttext.load_model('cc.cs.300.bin')
        if dim != 300:
            fasttext.util.reduce_model(self.ft, dim)

    def get_config(self):
        return {"dim": self._dim}

    def call(self, inputs, training):
        def fasttext_eager(inputs):
            return np.array([self.ft.get_word_vector(w) for w in inputs])
        inputs_ft = tf.numpy_function(func=fasttext_eager, inp=[inputs], Tout=tf.float32)
        inputs_ft = tf.ensure_shape(inputs_ft, (None, args.fasttext_dim))  # None for uncertain length of inputs
        return inputs_ft


# A layer setting given rate of elements to zero.
class MaskElements(tf.keras.layers.Layer):
    def __init__(self, rate):
        super().__init__()
        self._rate = rate

    def get_config(self):
        return {"rate": self._rate}

    def call(self, inputs, training):
        if training:
            # : Generate as many random uniform numbers in range [0, 1) as there are
            # values in `tf.RaggedTensor` `inputs` using a single `tf.random.uniform` call
            # (without setting seed in any way, so with just a single parameter `shape`).
            # Then, set the values in `inputs` to zero if the corresponding generated
            # random number is less than `self._rate`.
            # use tf.shape -> returns a now not known tensor (and different for each call)
            random = tf.random.uniform(shape=tf.shape(inputs.values))
            zero_and_ones = tf.cast(tf.math.greater_equal(random, tf.constant([self._rate])), tf.int64)
            inputs_masked = inputs.values * zero_and_ones
            return inputs.with_values(inputs_masked)
        else:
            return inputs


class Model(tf.keras.Model):

    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset, morph_analyzer: MorphoAnalyzer) -> None:

        # A: interpreting hyper parameters
        if args.l2:
            reg = tf.keras.regularizers.L2(args.l2)
        else:
            reg = None

        if not args.decay or args.decay in ["None", "none"]:
            learning_rate = args.learning_rate
        else:
            decay_steps = (train.size / args.batch_size) * args.epochs
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

        # B: build model

        # Implement a one-layer RNN network. The input `words` is
        # a RaggedTensor of strings, each batch example being a list of words.
        words = tf.keras.layers.Input(shape=[None], dtype=tf.string, ragged=True)

        # compute word embeddings
        if args.fasttext_mode in ['replace', 'concat']:
            words_flattened = words.values
            w_embeddings_fasttext = FastText(dim=args.fasttext_dim)(words_flattened)
            w_embeddings_fasttext = words.with_values(w_embeddings_fasttext)
        if args.fasttext_mode is None or args.fasttext_mode == 'concat':
            # (tagger_we): Map strings in `words` to indices by using the `word_mapping` of `train.forms`.
            words_indices = train.forms.word_mapping(words)
            words_indices = MaskElements(rate=args.word_masking)(words_indices)

            # (tagger_we): Embed input words with dimensionality `args.we_dim`. Note that the `word_mapping`
            # provides a `vocabulary_size()` call returning the number of unique words in the mapping.
            w_embeddings_classic = tf.keras.layers.Embedding(train.forms.word_mapping.vocabulary_size(),
                                                             args.we_dim)(words_indices)
        # combine word embeddings based on args.fasttext_mode
        if args.fasttext_mode == 'concat':
            w_embeddings = tf.keras.layers.Concatenate()([w_embeddings_fasttext, w_embeddings_classic])
        elif args.fasttext_mode == 'replace':
            w_embeddings = w_embeddings_fasttext
        elif args.fasttext_mode is None:
            w_embeddings = w_embeddings_classic
        else:
            raise NotImplementedError("Allowed values for 'args.fasttext_mode' are 'concat', 'replace' and None.")
        print(w_embeddings)

        # : Create a vector of input words from all batches using `words.values`
        # and pass it through `tf.unique`, obtaining a list of unique words and
        # indices of the original flattened words in the unique word list.
        unique_tensor = tf.unique(words.values)
        unique_words = unique_tensor.y
        flattened_words_indices = unique_tensor.idx

        # : Create sequences of letters by passing the unique words through
        # `tf.strings.unicode_split` call; use "UTF-8" as `input_encoding`.
        unique_word_letters = tf.strings.unicode_split(unique_words, input_encoding='UTF-8')

        # : Map the letters into ids by using `char_mapping` of `train.forms`.
        unique_word_letter_ids = train.forms.char_mapping(unique_word_letters)

        # : Embed the input characters with dimensionality `args.cle_dim`.
        cle_embeddings = tf.keras.layers.Embedding(train.forms.char_mapping.vocabulary_size(),
                                                   args.cle_dim)(unique_word_letter_ids)

        # : Pass the embedded letters through a bidirectional GRU layer
        # with dimensionality `args.cle_dim`, obtaining character-level representations
        # of the whole words, **concatenating** the outputs of the forward and backward RNNs.
        cle_forward_seq = tf.keras.layers.GRU(units=args.rnn_cell_dim, kernel_regularizer=reg)(cle_embeddings)
        cle_backward_seq = tf.keras.layers.GRU(units=args.rnn_cell_dim, kernel_regularizer=reg, go_backwards=True)\
            (cle_embeddings)
        cle_sequences = tf.keras.layers.Concatenate()([cle_forward_seq, cle_backward_seq])
        # cle_sequences = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=args.rnn_cell_dim),
        #                                               merge_mode='concat')(cle_embeddings)

        # : Use `tf.gather` with the indices generated by `tf.unique` to transform
        # the computed character-level representations of the unique words to representations
        # of the flattened (non-unique) words.
        flattened_words_cle = tf.gather(cle_sequences, flattened_words_indices)

        # : Then, convert these character-level word representations into
        # a RaggedTensor of the same shape as `words` using `words.with_values` call.
        word_cle_ragged = words.with_values(flattened_words_cle)

        # : Concatenate the word-level embeddings and the computed character-level WEs
        # (in this order).
        concatted_embeddings = tf.keras.layers.Concatenate()([w_embeddings, word_cle_ragged])

        # (tagger_we): Create the specified `args.rnn_cell` RNN cell (LSTM, GRU) with
        # dimension `args.rnn_cell_dim`. The cell should produce an output for every
        # sequence element (so a 3D output). Then apply it in a bidirectional way on
        # the word representations, **summing** the outputs of forward and backward RNNs.
        if args.rnn_cell == 'LSTM':
            cell_type = tf.keras.layers.LSTM
        elif args.rnn_cell == 'GRU':
            cell_type = tf.keras.layers.GRU
        else:
            raise NotImplementedError(f"{args.rnn_cell} is not a valid RNN cell")

        rnn_sequences = concatted_embeddings
        rnn_sequences_previous = None
        for i in range(args.rnn_layers):
            rnn_sequences = tf.keras.layers.Bidirectional(cell_type(units=args.rnn_cell_dim, return_sequences=True),
                                                           merge_mode='sum')(rnn_sequences)
            if args.rnn_residual == "every" and rnn_sequences_previous is not None:
                rnn_sequences = tf.keras.layers.Add(name=f"residual_connection_{i}")([rnn_sequences_previous, rnn_sequences])
            # set previous sequence for next loop
            rnn_sequences_previous = rnn_sequences

        if args.rnn_residual == "end":  # does not work because Received shapes (None, 96) and (None, 16)
            rnn_sequences = tf.keras.layers.Add(name="residual_connection")([concatted_embeddings, rnn_sequences])

        # (tagger_we): Add a softmax classification layer into as many classes as there are unique
        # tags in the `word_mapping` of `train.tags`. Note that the Dense layer can process
        # a RaggedTensor without any problem.
        predictions = tf.keras.layers.Dense(train.tags.word_mapping.vocabulary_size(), activation=tf.nn.softmax,
                                            kernel_regularizer=reg)(rnn_sequences)

        # Check that the created predictions are a 3D tensor.
        assert predictions.shape.rank == 3
        super().__init__(inputs=words, outputs=predictions)
        self.ragged_loss = lambda yt, yp: tf.losses.SparseCategoricalCrossentropy()(yt.values, yp.values)
        self.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                     loss=self.ragged_loss,
                     metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")])
        # self.summary()

        # define callbacks
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)
        self.best_checkpoint_path = os.path.join(args.logdir, "tagger_competition.ckpt")
        self.ckp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.best_checkpoint_path,
                                                               save_weights_only=False, monitor='val_accuracy',
                                                               mode='max', save_best_only=True)


def main(args):
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

    # Load the data. Using analyses is only optional.
    morpho = MorphoDataset("czech_pdt")
    analyses = MorphoAnalyzer("czech_pdt_analyses")

    def tagging_dataset(example):
        return example['forms'], morpho.train.tags.word_mapping(example['tags'])

    def create_dataset(name):
        dataset = getattr(morpho, name).dataset
        dataset = dataset.map(tagging_dataset)
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        return dataset

    train, dev, test = create_dataset("train"), create_dataset("dev"), create_dataset("test")

    # : Create the model and train it
    model = Model(args, morpho.train, analyses)
    print(args)
    model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[model.tb_callback, model.ckp_callback])
    print(args)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)

    # use best checkpoint to make predictions
    model = tf.keras.models.load_model(model.best_checkpoint_path,
                                       custom_objects={model.ragged_loss.__name__: model.ragged_loss})

    with open(os.path.join(args.logdir, "tagger_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # : Predict the tags on the test set; update the following prediction
        # command if you use other output structure than in tagger_we.
        predictions = model.predict(test)
        tag_strings = morpho.test.tags.word_mapping.get_vocabulary()
        for sentence in predictions:
            for word in sentence:
                print(tag_strings[np.argmax(word)], file=predictions_file)
            print(file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
