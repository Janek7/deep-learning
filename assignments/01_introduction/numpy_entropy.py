#!/usr/bin/env python3
import argparse

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_path", default="numpy_entropy_data_4.txt", type=str, help="Data distribution path.")
parser.add_argument("--model_path", default="numpy_entropy_model_4.txt", type=str, help="Model distribution path.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace) -> tuple[float, float, float]:
    # : Load data distribution, each line containing a datapoint -- a string.
    data_dict = {}
    with open(args.data_path, "r") as data:
        for line in data:
            line = line.rstrip("\n")
            # : Process the line, aggregating data with built-in Python
            # data structures (not NumPy, which is not suitable for incremental
            # addition and string mapping).
            if line in data_dict:
                data_dict[line] += 1
            else:
                data_dict[line] = 1
    total_count = sum([count for string, count in data_dict.items()])
    data_keys = list(data_dict.keys())

    # : Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. Alternatively,
    # the NumPy array might be created after loading the model distribution.
    data_distribution = np.zeros(len(data_dict), dtype=float)
    for i in range(len(data_keys)):
        data = data_keys[i]
        data_distribution[i] = data_dict[data] / total_count

    # : Load model distribution, each line `string \t probability`.
    model_dict = {}
    with open(args.model_path, "r") as model:
        for line in model:
            # : process the line, aggregating using Python data structures
            line = line.rstrip("\n").split("\t")
            model_dict[line[0]] = line[1]
    model_strings = [string for string, prob in model_dict.items()]
    # check for missing data
    for string in data_keys:
        if string not in model_strings:
            model_dict[string] = 0

    # : Create a NumPy array containing the model distribution.
    model_distribution = np.zeros(len(data_dict), dtype=float)
    for i in range(len(data_keys)):
        data = data_keys[i]
        model_distribution[i] = model_dict[data]

    # : Compute the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    entropy = - data_distribution.dot(np.log(data_distribution))

    # : Compute cross-entropy H(data distribution, model distribution).
    # When some data distribution elements are missing in the model distribution,
    # return `np.inf`.
    if 0 in model_distribution:
        crossentropy = np.inf
    else:
        crossentropy = - data_distribution.dot(np.log(model_distribution))

    # : Compute KL-divergence D_KL(data distribution, model_distribution),
    # again using `np.inf` when needed.
    kl_divergence = crossentropy - entropy

    # Return the computed values for ReCodEx to validate
    return entropy, crossentropy, kl_divergence

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, crossentropy, kl_divergence = main(args)
    print("Entropy: {:.2f} nats".format(entropy))
    print("Crossentropy: {:.2f} nats".format(crossentropy))
    print("KL divergence: {:.2f} nats".format(kl_divergence))
