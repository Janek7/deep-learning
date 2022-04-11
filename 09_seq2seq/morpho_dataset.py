from __future__ import annotations

import itertools
import os
import sys
from typing import BinaryIO, Optional, Sequence, TextIO
import urllib.request
import zipfile
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import tensorflow as tf


# Loads a morphological dataset in a vertical format.
# - The data consists of three Datasets
#   - train
#   - dev
#   - test
# - Each dataset is composed of
#   - size: number of sentences in the dataset
#   - forms, lemmas, tags: objects containing the following fields:
#     - strings: a Python list containing input sentences, each being
#         a list of strings (forms/lemmas/tags)
#     - word_mapping: a `tf.keras.layers.StringLookup` object capable of
#         mapping words to indices. It is constructed on the train set and
#         shared by the dev and test sets.
#     - char_mapping: a `tf.keras.layers.StringLookup` object capable of
#         mapping characters to indices. It is constructed on the train set
#         and shared by the dev and test sets.
#   - dataset: a `tf.data.Dataset` containing a dictionary with "forms", "lemmas", "tags".
class MorphoDataset:
    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/2122/datasets/"

    class Factor:
        BOW: int = 1
        EOW: int = 1
        word_mapping: tf.keras.layers.StringLookup
        char_mapping: tf.keras.layers.StringLookup

        def __init__(self) -> None:
            self.strings = []

        def finalize(self, train: Optional[Factor] = None, add_bow_eow: bool = False) -> None:
            # Create mappings
            if train:
                self.word_mapping = train.word_mapping
                self.char_mapping = train.char_mapping
            else:
                self.word_mapping = tf.keras.layers.StringLookup(
                    vocabulary=sorted(set(string for sentence in self.strings for string in sentence)))

                additional_characters = []
                if add_bow_eow:
                    additional_characters.extend(["[BOW]", "[EOW]"])
                self.char_mapping = tf.keras.layers.StringLookup(
                    vocabulary=additional_characters + sorted(set(
                        char for sentence in self.strings for string in sentence for char in string)))

    class Dataset:
        def __init__(self, data_file: BinaryIO, train: Optional[Dataset] = None,
                     max_sentences: Optional[int] = None, add_bow_eow: bool = False) -> None:
            # Create factors
            self._factors = (MorphoDataset.Factor(), MorphoDataset.Factor(), MorphoDataset.Factor())
            self._factors_tensors = None

            # Load the data
            self._size = 0
            in_sentence = False
            for line in data_file:
                line = line.decode("utf-8").rstrip("\r\n")
                if line:
                    if not in_sentence:
                        for factor in self._factors:
                            factor.strings.append([])
                        self._size += 1

                    columns = line.split("\t")
                    assert len(columns) == len(self._factors)
                    for column, factor in zip(columns, self._factors):
                        factor.strings[-1].append(column)

                    in_sentence = True
                else:
                    in_sentence = False
                    if max_sentences is not None and self._size >= max_sentences:
                        break

            # Finalize the mappings
            for i, factor in enumerate(self._factors):
                factor.finalize(train._factors[i] if train else None, add_bow_eow)

        @property
        def size(self) -> int:
            return self._size

        @property
        def forms(self) -> MorphoDataset.Factor:
            return self._factors[0]

        @property
        def lemmas(self) -> MorphoDataset.Factor:
            return self._factors[1]

        @property
        def tags(self) -> MorphoDataset.Factor:
            return self._factors[2]

        @property
        def dataset(self) -> tf.data.Dataset:
            if self._factors_tensors is None:
                self._factors_tensors = {
                    key: tf.RaggedTensor.from_row_lengths(list(itertools.chain(*v.strings)), list(map(len, v.strings)))
                    for key, v in [("forms", self.forms), ("lemmas", self.lemmas), ("tags", self.tags)]}
            return tf.data.Dataset.from_tensor_slices(self._factors_tensors)

    def __init__(self, dataset, max_sentences=None, add_bow_eow=False):

        path = "{}.zip".format(dataset)
        if not os.path.exists(path):
            print("Downloading dataset {}...".format(dataset), file=sys.stderr)
            urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename=path)

        with zipfile.ZipFile(path, "r") as zip_file:
            for dataset in ["train", "dev", "test"]:
                with zip_file.open("{}_{}.txt".format(os.path.splitext(path)[0], dataset), "r") as dataset_file:
                    setattr(self, dataset, self.Dataset(dataset_file,
                                                        train=self.train if dataset != "train" else None,
                                                        max_sentences=max_sentences, add_bow_eow=add_bow_eow))

    train: Dataset
    dev: Dataset
    test: Dataset

    # Evaluation infrastructure.
    @staticmethod
    def evaluate(gold_dataset: MorphoDataset.Factor, predictions: Sequence[str]) -> float:
        gold_sentences = gold_dataset.strings

        predicted_sentences, in_sentence = [], False
        for line in predictions:
            line = line.rstrip("\n")
            if not line:
                in_sentence = False
            else:
                if not in_sentence:
                    predicted_sentences.append([])
                    in_sentence = True
                predicted_sentences[-1].append(line)

        if len(predicted_sentences) != len(gold_sentences):
            raise RuntimeError("The predictions contain different number of sentences than gold data: {} vs {}".format(
                len(predicted_sentences), len(gold_sentences)))

        correct, total = 0, 0
        for i, (predicted_sentence, gold_sentence) in enumerate(zip(predicted_sentences, gold_sentences)):
            if len(predicted_sentence) != len(gold_sentence):
                raise RuntimeError("Predicted sentence {} has different number of words than gold: {} vs {}".format(
                    i + 1, len(predicted_sentence), len(gold_sentence)))
            correct += sum(predicted == gold for predicted, gold in zip(predicted_sentence, gold_sentence))
            total += len(predicted_sentence)

        return 100 * correct / total

    @staticmethod
    def evaluate_file(gold_dataset: MorphoDataset.Factor, predictions_file: TextIO) -> float:
        predictions = predictions_file.readlines()
        return MorphoDataset.evaluate(gold_dataset, predictions)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", default=None, type=str, help="Prediction file to evaluate")
    parser.add_argument("--corpus", default="czech_pdt", type=str, help="The corpus to evaluate")
    parser.add_argument("--dataset", default="dev", type=str, help="The dataset to evaluate (dev/test)")
    parser.add_argument("--task", default="tagger", type=str, help="Task to evaluate (tagger/lemmatizer)")
    args = parser.parse_args()

    if args.evaluate:
        gold = getattr(MorphoDataset(args.corpus), args.dataset)
        if args.task == "tagger":
            gold = gold.tags
        elif args.task == "lemmatizer":
            gold = gold.lemmas
        else:
            raise ValueError("Unknown task '{}', valid values are only 'tagger' or 'lemmatizer'".format(args.task))

        with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
            accuracy = MorphoDataset.evaluate_file(gold, predictions_file)
        print("{} accuracy: {:.2f}%".format(args.task.title(), accuracy))
