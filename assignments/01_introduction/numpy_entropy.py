#!/usr/bin/env python3
import argparse
from math import log

import numpy as np
import scipy.stats

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_path", default="numpy_entropy_data_1.txt", type=str, help="Data distribution path.")
parser.add_argument("--model_path", default="numpy_entropy_model_1.txt", type=str, help="Model distribution path.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace) -> tuple[float, float, float]:
    # TODO: Load data distribution, each line containing a datapoint -- a string.
    string_counts = {}
    with open(args.data_path, "r") as data:
        for line in data:
            line = line.rstrip("\n")
            # TODO: Process the line, aggregating data with built-in Python
            # data structures (not NumPy, which is not suitable for incremental
            # addition and string mapping).
            if line in string_counts:
                string_counts[line] += 1
            else:
                string_counts[line] = 1
    string_counts = list(sorted([(string, count) for string, count in string_counts.items()], key=lambda t: t[0]))
    data = [string for string, count in string_counts]
    print(string_counts)

    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. Alternatively,
    # the NumPy array might be created after loading the model distribution.
    total_count = sum([count for string, count in string_counts])
    data_distribution = np.array([count / total_count for string, count in string_counts])

    # TODO: Load model distribution, each line `string \t probability`.
    lines = []
    with open(args.model_path, "r") as model:
        for line in model:
            line = line.rstrip("\n")
            lines.append(line.split("\t"))
            # TODO: process the line, aggregating using Python data structures
    lines.sort(key=lambda line: line[0])

    # TODO: Create a NumPy array containing the model distribution.
    model_distribution = np.array([float(line[1]) for line in lines])
    model_data = [line[0] for line in lines]
    print(model_distribution)

    # TODO: Compute the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    entropy = - data_distribution.dot(np.log(data_distribution))

    # TODO: Compute cross-entropy H(data distribution, model distribution).
    # When some data distribution elements are missing in the model distribution,
    # return `np.inf`.
    if set(model_data) != set(data):
        crossentropy = np.inf
    else:
        crossentropy = - data_distribution.dot(np.log(model_distribution))

    # TODO: Compute KL-divergence D_KL(data distribution, model_distribution),
    # again using `np.inf` when needed.
    if set(model_data) != set(data):
        kl_divergence = np.inf
    else:
        kl_divergence = crossentropy - entropy

    # Return the computed values for ReCodEx to validate
    return entropy, crossentropy, kl_divergence

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, crossentropy, kl_divergence = main(args)
    print("Entropy: {:.2f} nats".format(entropy))
    print("Crossentropy: {:.2f} nats".format(crossentropy))
    print("KL divergence: {:.2f} nats".format(kl_divergence))
