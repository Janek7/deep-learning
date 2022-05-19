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
parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing.")
parser.add_argument("--warmup_epochs", default=1, type=float, help="Number of warmup epochs.")


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
    # robeczech = transformers.TFAutoModel.from_pretrained("ufal/robeczech-base")

    # Load the data.
    reading_comprehension_dataset = ReadingComprehensionDataset()
    debug = False

    def create_dataset(name) -> tf.data.Dataset:
        dataset = getattr(reading_comprehension_dataset, name)

        # 1) tokenize data
        context_question_pairs, answer_starts, answer_ends = [], [], []
        for paragraph in dataset.paragraphs:  # iterate over all paragraphs
            for qa_pair in paragraph["qas"]:  # iterate over all questions in a paragraph
                encoded_context_question_pair = tokenizer(paragraph["context"], qa_pair["question"], max_length=512,
                                                          truncation="only_first", padding="max_length")
                # if dataset is test, append only to context_question_pairs, answers are empty
                if name == 'test':
                    context_question_pairs.append(encoded_context_question_pair["input_ids"])
                else:
                    for answer in qa_pair["answers"]:  # iterate over all answers of a question
                        # compute answer end index
                        last_answer_word = answer["text"].split(" ")[-1]
                        answer_end_idx = paragraph["context"][answer["start"]:].find(last_answer_word)
                        answer_end_idx += answer["start"]
                        answer_end_token_idx = encoded_context_question_pair.char_to_token(answer["start"])
                        if debug:
                            print(paragraph["context"])
                            print(encoded_context_question_pair["input_ids"])
                            print(answer["text"], tokenizer(answer["text"])["input_ids"])
                            print("start idx", answer["start"], "token:", encoded_context_question_pair.char_to_token(answer["start"]))
                            print(last_answer_word)
                            print("end idx", answer_end_idx, "token:", encoded_context_question_pair.char_to_token(answer_end_idx))
                            print("-------------")

                        # only if computation was successful, append whole triple (fails in 22 cases for train)
                        if answer_end_token_idx is not None:
                            # A) record for every answer the same tokenized context/question
                            context_question_pairs.append(encoded_context_question_pair["input_ids"])
                            # B) record for every answer the respective token IDs of start and end words
                            answer_starts.append(encoded_context_question_pair.char_to_token(answer["start"]))
                            answer_ends.append(answer_end_token_idx)
        # convert lists to tensors
        context_question_pairs = tf.constant(context_question_pairs)
        answer_starts = tf.constant(answer_starts)
        answer_ends = tf.constant(answer_ends)

        # 2) create tf.data.DataSet
        if name != 'test':
            dataset = tf.data.Dataset.from_tensor_slices((context_question_pairs, answer_starts, answer_ends))
            # combine answer_start and answer_end to one output dict per sample
            def one_output_map(context_question_pair, answer_start, answer_end):
                return context_question_pair, {"answer_start": answer_start, "answer_end": answer_end}
            dataset = dataset.map(one_output_map)
        else:
            dataset = tf.data.Dataset.from_tensor_slices((context_question_pairs))
        if name == "train":
            dataset = dataset.shuffle(buffer_size=10000, seed=args.seed)
        print(f"{name} dataset: {len(dataset)}")
        dataset = dataset.batch(args.batch_size)
        return dataset

    print("create datasets")
    train = create_dataset("train")
    dev = create_dataset("dev")
    test = create_dataset("test")

    # TODO: Create the model and train it
    model = ...

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "reading_comprehension.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the answers as strings, one per line.
        predictions = ...

        for answer in predictions:
            print(answer, file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
