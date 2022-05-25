#!/usr/bin/env python3
# TEAM MEMBERS:
# Antonio Krizmanic - 2b193238-8e3c-11ec-986f-f39926f24a9c
# Janek Putz - e31a3cae-8e6c-11ec-986f-f39926f24a9c
import argparse
import datetime
import os
import re
from typing import List

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
import transformers

from reading_comprehension_dataset import ReadingComprehensionDataset

# : Define reasonable defaults and optionally more parameters.
# Also, you can set the number of the threads 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--epochs", default=4, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# additional parameter
parser.add_argument("--decay", default="None", type=str, help="Learning decay rate type")
parser.add_argument("--learning_rate", default=2e-5, type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default=1e-5, type=float, help="Final learning rate.")
parser.add_argument("--dropout", default=0, type=float, help="Dropout")
parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing.")
parser.add_argument("--warmup_epochs", default=1, type=float, help="Number of warmup epochs.")


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
    def __init__(self, args, robeczech, train):

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

        # create warmup
        warmup_steps = int(len(train) * args.warmup_epochs)  # len(train) -> number of steps in one epoch
        learning_rate = LinearWarmup(warmup_steps, following_schedule=learning_rate)

        # B) ARCHITECTURE
        inputs = {
            "input_ids": tf.keras.layers.Input(shape=[None], dtype=tf.int32, ragged=True),
            "attention_mask": tf.keras.layers.Input(shape=[None], dtype=tf.int32, ragged=True)
        }

        robeczech_output = robeczech(input_ids=inputs["input_ids"].to_tensor(),
                                     attention_mask=inputs["attention_mask"].to_tensor()).last_hidden_state
        robeczech_dropout = tf.keras.layers.Dropout(args.dropout)(robeczech_output)

        # [:, :, 0] removes the 3rd dimension and pulls the value in the second dimension (alternative to reshape)
        # alternative: [..., 0] -> ... is placeholder for arbitrary number of dimensions
        answer_start_output = tf.keras.layers.Dense(1)(robeczech_dropout)[:, :, 0]
        answer_start_output_softmax = tf.keras.layers.Softmax(name="answer_start")(answer_start_output)
        answer_end_output = tf.keras.layers.Dense(1)(robeczech_dropout)[:, :, 0]
        answer_end_output_softmax = tf.keras.layers.Softmax(name="answer_end")(answer_end_output)
        outputs = {"answer_start": answer_start_output_softmax, "answer_end": answer_end_output_softmax}

        super().__init__(inputs=inputs, outputs=outputs)

        # C) COMPILE
        self.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                     loss={"answer_start": tf.keras.losses.SparseCategoricalCrossentropy(),
                           "answer_end": tf.keras.losses.SparseCategoricalCrossentropy()},
                     metrics={
                         "answer_start": [tf.metrics.SparseCategoricalAccuracy(name="accuracy_start")],
                         "answer_end": [tf.metrics.SparseCategoricalAccuracy(name="accuracy_end")]
                     })
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
    tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/robeczech-base")
    robeczech = transformers.TFAutoModel.from_pretrained("ufal/robeczech-base")

    # Load the data.
    reading_comprehension_dataset = ReadingComprehensionDataset()

    def create_dataset(name) -> (tf.data.Dataset, List[transformers.BatchEncoding], List[str]):
        # returns a tf.data.Dataset. In case of 'test', return additionaly list of BatchEncodings of context, question
        # pairs and list of original raw string contexts
        #
        # Note: to improve it, before adding triples to result, check if computed answer end token index is in the
        # range of the paragraph/context in the tokenized pair of context and question (otherwise we would point in the
        # question -> train with false examples)
        dataset = getattr(reading_comprehension_dataset, name)

        # 1) tokenize data
        context_question_pair_tokens, context_question_pair_attention_masks, answer_starts, answer_ends = [], [], [], []
        encoded_context_question_pairs, contexts_raw = [], []
        for paragraph in dataset.paragraphs:  # iterate over all paragraphs
            for qa_pair in paragraph["qas"]:  # iterate over all questions in a paragraph
                encoded_context_question_pair = tokenizer(paragraph["context"], qa_pair["question"], max_length=512,
                                                          truncation="only_first")
                # if dataset is test, append only to context_question_pairs, answers are empty
                if name == 'test':
                    context_question_pair_tokens.append(encoded_context_question_pair["input_ids"])
                    context_question_pair_attention_masks.append(encoded_context_question_pair["attention_mask"])
                    # record raw and tokenized contexts for inference
                    contexts_raw.append(paragraph["context"])
                    encoded_context_question_pairs.append(encoded_context_question_pair)
                else:
                    for answer in qa_pair["answers"]:  # iterate over all answers of a question
                        # compute answer end index
                        answer_end_token_idx = encoded_context_question_pair.char_to_token(
                            answer["start"] + len(answer["text"]) - 1)
                        # only if computation was successful, append whole triple (fails in 22 cases for train)
                        if answer_end_token_idx is not None:
                            # A) record for every answer the same tokenized context/question
                            context_question_pair_tokens.append(encoded_context_question_pair["input_ids"])
                            context_question_pair_attention_masks.append(
                                encoded_context_question_pair["attention_mask"])
                            # B) record for every answer the respective token IDs of start and end words
                            answer_starts.append(encoded_context_question_pair.char_to_token(answer["start"]))
                            answer_ends.append(answer_end_token_idx)

        # convert lists to tensors
        context_question_pair_tokens = tf.ragged.constant(context_question_pair_tokens)
        context_question_pair_attention_masks = tf.ragged.constant(context_question_pair_attention_masks)
        answer_starts = tf.constant(answer_starts)
        answer_ends = tf.constant(answer_ends)

        # 2) create tf.data.DataSet
        if name != 'test':
            dataset = tf.data.Dataset.from_tensor_slices((context_question_pair_tokens,
                                                          context_question_pair_attention_masks,
                                                          answer_starts,
                                                          answer_ends))
            # combine answer_start/tokens and answer_end/attention_mask to one output/input dict per sample
            def map(context_question_pair_tokens, context_question_pair_attention_masks, answer_start, answer_end):
                return {"input_ids": context_question_pair_tokens,
                        "attention_mask": context_question_pair_attention_masks}, \
                       {"answer_start": answer_start, "answer_end": answer_end}
            dataset = dataset.map(map)
        else:
            dataset = tf.data.Dataset.from_tensor_slices((context_question_pair_tokens,
                                                          context_question_pair_attention_masks))
            # combine tokens and attention_mask to one input dict per sample
            def map(context_question_pair_tokens, context_question_pair_attention_masks):
                return {"input_ids": context_question_pair_tokens,
                        "attention_mask": context_question_pair_attention_masks}
            dataset = dataset.map(map)

        if name == "train":
            dataset = dataset.shuffle(buffer_size=10000, seed=args.seed)

        print(f"{name} dataset: {len(dataset)}")
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args.batch_size))
        return dataset, encoded_context_question_pairs, contexts_raw

    print("create datasets")
    train, _, _ = create_dataset("train")
    dev, _, _ = create_dataset("dev")
    test, test_contexts_tokenized, test_contexts = create_dataset("test")

    # : Create the model and train it
    model = Model(args, robeczech, train)

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
    with open(os.path.join(args.logdir, "reading_comprehension.txt"), "w", encoding="utf-8") as predictions_file:
        # : Predict the answers as strings, one per line.
        n = 0
        for test_batch in test:  # predict simple batches because of different max lengths in batches
            predictions = model.predict(test_batch)
            predictions_starts = predictions["answer_start"]
            predictions_ends = predictions["answer_end"]

            for batch_sample_i in range(len(predictions_ends)):
                start_token_idx = tf.argmax(predictions_starts[batch_sample_i]).numpy()
                # [start_token_idx:] is necessary to not retrieve smaller indices for end than for start token
                end_token_idx = start_token_idx + tf.argmax(predictions_ends[batch_sample_i][start_token_idx:]).numpy()
                # print(start_token_idx, end_token_idx)
                try:
                    answer = test_contexts[n][test_contexts_tokenized[n].token_to_chars(start_token_idx).start:
                                              test_contexts_tokenized[n].token_to_chars(end_token_idx - 1).end]
                except Exception as ex:
                    print(ex)
                    answer = "answer lookup failed"  # dummy answer
                n += 1
                print(answer, file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
