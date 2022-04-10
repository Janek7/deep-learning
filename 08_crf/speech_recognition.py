#!/usr/bin/env python3
# TEAM MEMBERS:
# Antonio Krizmanic - 2b193238-8e3c-11ec-986f-f39926f24a9c
# Janek Putz - e31a3cae-8e6c-11ec-986f-f39926f24a9c
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from common_voice_cs import CommonVoiceCs

# : Define reasonable defaults and optionally more parameters
# standard params
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=2, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# architecture params
parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
parser.add_argument("--rnn_cell_dim", default=16, type=int, help="RNN cell dimension.")
parser.add_argument("--rnn_bidirectional_merge", default="sum", type=str, help="Bidirectional merge mode")
parser.add_argument("--rnn_layers", default=1, type=int, help="Number of bidirectional rnn layers behind each other.")
parser.add_argument("--rnn_residual", default=None, type=str, help="where to create a residual connection.")
# regularization params
parser.add_argument("--batch_norm", default=True, type=bool, help="Batch norm after input")
parser.add_argument("--l2", default=0.00, type=float, help="L2 regularization.")
parser.add_argument("--decay", default="cosine", type=str, help="Learning decay rate type")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default=0.0001, type=float, help="Final learning rate.")


class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace, train: tf.data.Dataset) -> None:
        # Use where?
        # self._ctc_beam = args.ctc_beam

        # A) REGULARIZATION PARAMS
        reg = tf.keras.regularizers.L2(args.l2) if args.l2 else None

        if not args.decay or args.decay in ["None", "none"]:
            learning_rate = args.learning_rate
        else:
            decay_steps = len(train) * args.epochs
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

        # B) COMPOSE MODEL
        inputs = tf.keras.layers.Input(shape=[None, CommonVoiceCs.MFCC_DIM], dtype=tf.float32, ragged=True)
        # if args.batch_norm:
        #    inputs = tf.keras.layers.BatchNormalization()(input)

        # : Create a suitable model. You should:
        # - use a bidirectional RNN layer(s) to contextualize the input sequences.
        # - optionally use suitable regularization
        if args.rnn_cell == 'LSTM':
            cell_type = tf.keras.layers.LSTM
        elif args.rnn_cell == 'GRU':
            cell_type = tf.keras.layers.GRU
        else:
            raise NotImplementedError(f"{args.rnn_cell} is not a valid RNN cell")

        rnn_sequences = inputs
        rnn_sequences_previous = None
        for i in range(args.rnn_layers):
            rnn_sequences = tf.keras.layers.Bidirectional(cell_type(units=args.rnn_cell_dim, return_sequences=True,
                                                                    kernel_regularizer=reg),
                                                          merge_mode=args.rnn_bidirectional_merge)(rnn_sequences)
            if args.rnn_residual == "every" and rnn_sequences_previous is not None:
                rnn_sequences = tf.keras.layers.Add(name=f"residual_connection_{i}")(
                    [rnn_sequences_previous, rnn_sequences])
            # set previous sequence for next loop
            rnn_sequences_previous = rnn_sequences

        if args.rnn_residual == "end":  # does not work because of different shapes
            rnn_sequences = tf.keras.layers.Add(name="residual_connection")([inputs, rnn_sequences])

        #  - and finally generate logits for CRC loss/prediction as RaggedTensors.
        #   The logits should be generated by a dense layer with `1 + len(CommonVoiceCs.LETTERS)`
        #   outputs (the plus one is for the CTC blank symbol). Note that no
        #   activation should be used (the CTC operations will take care of it).
        logits = tf.keras.layers.Dense(units=1 + len(CommonVoiceCs.LETTERS), activation=None,
                                       kernel_regularizer=reg)(rnn_sequences)

        super().__init__(inputs=inputs, outputs=logits)

        # C) COMPILE MODEL

        # We compile the model with the CTC loss and EditDistance metric.
        # the `selt.ctc_loss` method.
        ed_metric_name = "accuracy"
        self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                     loss=self.ctc_loss,
                     # has to be named accuracy because of checkpoint call back requires somehow
                     metrics=[CommonVoiceCs.EditDistanceMetric(name=ed_metric_name)])

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)
        self.best_checkpoint_path = os.path.join(args.logdir, "tagger_competition.ckpt")
        self.ckp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.best_checkpoint_path,
                                                               save_weights_only=False, monitor=f'val_{ed_metric_name}',
                                                               mode='max', save_best_only=True)

    def ctc_loss(self, gold_labels: tf.RaggedTensor, logits: tf.RaggedTensor) -> tf.Tensor:
        assert isinstance(gold_labels, tf.RaggedTensor), "Gold labels given to CTC loss must be RaggedTensors"
        assert isinstance(logits, tf.RaggedTensor), "Logits given to CTC loss must be RaggedTensors"

        # : Use tf.nn.ctc_loss to compute the CTC loss.
        # - Convert the gold_labels to SparseTensor and pass `None` as `label_length`.
        # - Convert `logits` to a dense Tensor and then either transpose the
        #   logits to `[max_audio_length, batch, dim]` or set `logits_time_major=False`
        # - Use `logits.row_lengths()` method to obtain the `logit_length`
        # - Use the last class (the one with the highest index) as the `blank_index`.
        #
        # The `tc.nn.ctc_loss` returns a value for a single batch example, so average
        # them to produce a single value and return it.
        gold_labels_sparse = tf.cast(gold_labels.to_sparse(), tf.int32)
        logits_dense = logits.to_tensor()
        logit_length = tf.cast(logits.row_lengths(), tf.int32)
        result = tf.nn.ctc_loss(labels=gold_labels_sparse, logits=logits_dense,
                                label_length=None, logit_length=logit_length,
                                logits_time_major=False,
                                # TODO: set correctly?
                                blank_index=len(CommonVoiceCs.LETTERS))
        avg_result = tf.math.reduce_mean(-result)
        return avg_result

    def ctc_decode(self, logits: tf.RaggedTensor) -> tf.RaggedTensor:
        assert isinstance(logits, tf.RaggedTensor), "Logits given to CTC predict must be RaggedTensors"

        # : Run `tf.nn.ctc_greedy_decoder` or `tf.nn.ctc_beam_search_decoder`
        # to perform prediction.
        # - Convert the `logits` to a dense Tensor and then transpose them
        #   to shape `[max_audio_length, batch, dim]` using `tf.transpose`
        # - Use `logits.row_lengths()` method to obtain the `sequence_length`
        # - Convert the result of the decoded from a SparseTensor to a RaggedTensor
        decoded, _ = tf.nn.ctc_greedy_decoder(inputs=tf.transpose(logits.to_tensor(), [1, 0, 2]),
                                              sequence_length=tf.cast(logits.row_lengths(), tf.int32))
        predictions = tf.RaggedTensor.from_sparse(decoded[0])
        assert isinstance(predictions, tf.RaggedTensor), "CTC predictions must be RaggedTensors"
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

    # We override `predict_step` to run CTC decoding during prediction.
    def predict_step(self, data):
        data = data[0] if isinstance(data, tuple) else data
        y_pred = self(data, training=False)
        y_pred = self.ctc_decode(y_pred)
        return y_pred

    # We override `test_step` to run CTC decoding during evaluation.
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compute_loss(x, y, y_pred)
        y_pred = self.ctc_decode(y_pred)
        return self.compute_metrics(x, y, y_pred, None)


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

    # Load the data. Using analyses is only optional.
    cvcs = CommonVoiceCs()

    # Create input data pipeline.
    def create_dataset(name):
        def prepare_example(example):
            # : Create suitable batch examples.
            # - example["mfccs"] should be used as input
            # - the example["sentence"] is a UTF-8-encoded string with the target sentence
            #   - split it to unicode characters by using `tf.strings.unicode_split`
            #   - then pass it through the `cvcs.letters_mapping` layer to map
            #     the unicode characters to ids
            return example["mfccs"], cvcs.letters_mapping(tf.strings.unicode_split(example["sentence"], 'UTF-8'))

        dataset = getattr(cvcs, name).map(prepare_example)
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    train, dev, test = create_dataset("train"), create_dataset("dev"), create_dataset("test")

    # : Create the model and train it
    model = Model(args, train)

    best_checkpoint_path = "C:\\Users\\janek\\Development\\Git\\Prag\\deep-learning-lecture\\08_crf\\logs\\speech_recognition.py-2022-04-10_153538-bn=True,bs=10,d=cosine,e=2,l=0.0,lr=0.001,lrf=0.0001,rbm=sum,rc=LSTM,rcd=16,rl=1,rr=None,s=42,t=1\\speech_recognition.ckpt"
    # best_checkpoint_path = os.path.join(args.logdir, "speech_recognition.ckpt")
    # model.fit(
    #     train.take(10), batch_size=args.batch_size, epochs=args.epochs, validation_data=dev.take(10),
    #     callbacks=[tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0),
    #                tf.keras.callbacks.ModelCheckpoint(filepath=best_checkpoint_path,
    #                                                   save_weights_only=False, monitor='val_accuracy',
    #                                                   mode='max', save_best_only=True)]
    # )

    try:
        custom_objects = {
            CommonVoiceCs.EditDistanceMetric.__name__: CommonVoiceCs.EditDistanceMetric,
            model.ctc_decode.__name__: model.ctc_decode,
            model.ctc_loss.__name__: model.ctc_loss
        }
        best_model = tf.keras.models.load_model(best_checkpoint_path, custom_objects=custom_objects)
    except OSError:
        best_model = model

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "speech_recognition.txt"), "w", encoding="utf-8") as predictions_file:
        predictions = best_model.predict(test.take(1))
        for sentence in predictions:
            # sentence: eager tensor with shape [chars, LETTERS] -> probability of LETTERS for each char
            # -> get arg max of each char probability distribution? TODO: negative values possible?
            sentence_max = tf.math.argmax(sentence, axis=1)
            # print(max(sentence_max))
            # TODO: how to CTC blank symbol (+1) in lookup?
            print("".join(CommonVoiceCs.LETTERS[char] for char in sentence_max), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
