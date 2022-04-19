import os
import sys
from typing import Dict, List, Sequence, TextIO
import urllib.request

import numpy as np
import tensorflow as tf


class ModelNet:
    # The D, H, W are set in the constructor depending
    # on requested resolution and are only instance variables.
    D: int
    H: int
    W: int
    C: int = 1
    LABELS: List[str] = [
        "bathtub", "bed", "chair", "desk", "dresser",
        "monitor", "night_stand", "sofa", "table", "toilet",
    ]

    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/2122/datasets/modelnet{}.npz"

    class Dataset:
        def __init__(self, data: Dict[str, np.ndarray], seed: int = 42) -> None:
            self._data = data
            self._size = len(self._data["voxels"])

        @property
        def data(self) -> Dict[str, np.ndarray]:
            return self._data

        @property
        def size(self) -> int:
            return self._size

        @property
        def dataset(self) -> tf.data.Dataset:
            return tf.data.Dataset.from_tensor_slices(self._data)

    # The resolution parameter can be either 20 or 32.
    def __init__(self, resolution: int) -> None:
        assert resolution in [20, 32], "Only 20 or 32 resolution is supported"

        self.D = self.H = self.W = resolution
        url = self._URL.format(resolution)

        path = os.path.basename(url)
        if not os.path.exists(path):
            print("Downloading {} dataset...".format(path), file=sys.stderr)
            urllib.request.urlretrieve(url, filename=path)

        modelnet = np.load(path)
        for dataset in ["train", "dev", "test"]:
            data = dict((key[len(dataset) + 1:], modelnet[key]) for key in modelnet if key.startswith(dataset))
            setattr(self, dataset, self.Dataset(data))

    train: Dataset
    dev: Dataset
    test: Dataset

    # Evaluation infrastructure.
    @staticmethod
    def evaluate(gold_dataset: Dataset, predictions: Sequence[int]) -> float:
        gold = gold_dataset.data["labels"]

        if len(predictions) != len(gold):
            raise RuntimeError("The predictions are of different size than gold data: {} vs {}".format(
                len(predictions), len(gold)))

        correct = sum(gold[i] == predictions[i] for i in range(len(gold)))
        return 100 * correct / len(gold)

    @staticmethod
    def evaluate_file(gold_dataset: Dataset, predictions_file: TextIO) -> float:
        predictions = [int(line) for line in predictions_file]
        return ModelNet.evaluate(gold_dataset, predictions)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", default=None, type=str, help="Prediction file to evaluate")
    parser.add_argument("--dataset", default="dev", type=str, help="Gold dataset to evaluate")
    parser.add_argument("--dim", default=20, type=int, help="ModelNet dimensionality to use")
    args = parser.parse_args()

    if args.evaluate:
        with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
            accuracy = ModelNet.evaluate_file(getattr(ModelNet(args.dim), args.dataset), predictions_file)
        print("ModelNet accuracy: {:.2f}%".format(accuracy))
