#!/usr/bin/env python3
# TEAM MEMBERS:
# Antonio Krizmanic - 2b193238-8e3c-11ec-986f-f39926f24a9c
# Janek Putz - e31a3cae-8e6c-11ec-986f-f39926f24a9c
import argparse
import datetime
import logging
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
logger = logging.getLogger('SVHN')

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import bboxes_utils
import efficient_net
from svhn_dataset import SVHN

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--logging_level", default="info", type=str, help="Logging level")
parser.add_argument("--fine_tuning", default=False, type=bool, help="Optionally fine tune the efficient net core.")
parser.add_argument("--image_size", default=112, type=int, help="Width and height to resize image to uniform size.")
parser.add_argument("--iou_threshold", default=0.5, type=float, help="Threshold to assign anchors to gold bboxes.")


def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data
    svhn = SVHN()

    # create anchors
    # https://www.tensorflow.org/api_docs/python/tf/image/generate_bounding_box_proposals
    # https://www.tensorflow.org/api_docs/python/tf/image/draw_bounding_boxes
    anchors = np.array([[-1, -1, -1, -1]])
    for T in range(0, 85, 14):
        for L in range(0, 99, 7):
            anchors = np.append(anchors, [[T, L, T + 28, L + 14]], axis=0)
    anchors = np.delete(anchors, 0, 0)

    def create_dataset(dataset: tf.data.Dataset, training: bool) -> tf.data.Dataset:

        def prepare_data(example):
            example["classes"] = tf.cast(example["classes"], dtype=tf.int32)
            anchor_classes, anchor_bboxes = tf.numpy_function(
                bboxes_utils.bboxes_training, [anchors, example["classes"], example["bboxes"], 0.5],
                (tf.int32, tf.float32))
            output = {
                "classes": tf.ensure_shape(anchor_classes, [len(anchors)]),
                "bboxes": tf.ensure_shape(anchor_bboxes, [len(anchors), 4])
            }
            resized_image = tf.image.resize(example["image"], [args.image_size, args.image_size])
            return resized_image, output

        dataset = dataset.map(prepare_data)
        if training:
            dataset = dataset.shuffle(buffer_size=10000, seed=args.seed)
        dataset = dataset.batch(args.batch_size)

        return dataset

    train = create_dataset(svhn.train, True)
    for b in train:
        print(b)
        break
    exit()

    dev = create_dataset(svhn.dev, False)
    test = create_dataset(svhn.test, False)

    # Load the EfficientNet-B0 model
    efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False)
    efficientnet_b0.trainable = args.fine_tuning
    print(efficientnet_b0.outputs)

    # Start with retinanet like single stage detector
    # TODO: Create the model and train it
    inputs = None
    outputs = {
        "class": None,
        "bbox": None
    }
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss={  # keys fit to output dict
            "class": tf.keras.losses.SparseCategoricalCrossentropy(),
            "bbox": tfa.losses.SigmoidFocalCrossEntropy(alpha=0.25, gamma=2)
        },
        metrics={
            "class": [tf.keras.metrics.SparseCategoricalAccuracy("accuracy")],
            "bbox": []  # TODO
        }
    )

    model.summary()
    exit()

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "svhn_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the digits and their bounding boxes on the test set.
        # Assume that for a single test image we get
        # - `predicted_classes`: a 1D array with the predicted digits,
        # - `predicted_bboxes`: a [len(predicted_classes), 4] array with bboxes;
        for predicted_classes, predicted_bboxes in ...:
            output = []
            for label, bbox in zip(predicted_classes, predicted_bboxes):
                output += [label] + list(bbox)
            print(*output, file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    if args.logging_level == "info":
        logging.basicConfig(level=logging.INFO)
    elif args.logging_level == "debug":
        logging.basicConfig(level=logging.DEBUG)
    elif args.logging_level == "warning":
        logging.basicConfig(level=logging.WARNING)
    else:
        raise NotImplementedError("Use 'info' or 'debug' as logging level")

    main(args)
