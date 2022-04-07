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
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import bboxes_utils
import efficient_net
from svhn_dataset import SVHN

# : Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=2, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

parser.add_argument("--fine_tuning", default=False, type=bool, help="Optionally fine tune the efficient net core.")
parser.add_argument("--level", default=4, type=bool, help="Level of pyramid of efficient net to use as base.")
parser.add_argument("--image_size", default=224, type=int, help="Width and height to resize image to uniform size.")
parser.add_argument("--conv_filters", default=256, type=int, help="Number of filters in conv layers in heads.")
parser.add_argument("--iou_threshold", default=0.5, type=float, help="Threshold to assign anchors to gold bboxes.")
parser.add_argument("--iou_prediction", default=0.5, type=float, help="Threshold for non max suppresion.")
parser.add_argument("--score_threshold", default=0.2, type=float, help="Score threshold for non max suppresion.")

parser.add_argument("--batch_norm", default=True, type=bool, help="Batch normalization of conv. layers.")
parser.add_argument("--dropout", default=0.5, type=float, help="Dropout rate after efficient net layer.")
parser.add_argument("--l2", default=0.00, type=float, help="L2 regularization.")
parser.add_argument("--decay", default="cosine", type=str, help="Learning decay rate type")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default=0.0001, type=float, help="Final learning rate.")


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

    # 1) create anchors
    def anchors_new():
        anchors = np.array([[-1, -1, -1, -1]])
        square_anchor_size = 2 ** args.level * 4  # for vertical boxes use 2^l*4 for height and 2^l*2 for width
        for row in range(2 ** args.level, 2 ** args.level * 14 + 1, 2 ** args.level):
            for col in range(2 ** args.level, 2 ** args.level * 14 + 1, 2 ** args.level):
                anchors = np.append(anchors, [[row - square_anchor_size / 2,
                                               col - square_anchor_size / 2,
                                               row + square_anchor_size / 2,
                                               col + square_anchor_size / 2]], axis=0)
        anchors = np.delete(anchors, 0, 0)
        print("anchors:", anchors.shape)
        # print(anchors)
        return anchors

    anchors = anchors_new()

    # 2) Load the data
    svhn = SVHN()

    def create_dataset(dataset: tf.data.Dataset, training: bool) -> tf.data.Dataset:

        def prepare_data(example):
            example["classes"] = tf.cast(example["classes"], dtype=tf.int32)
            example["bboxes"] = example["bboxes"] / tf.cast(tf.shape(example["image"])[0], tf.float32)
            resized_image = tf.image.resize(example["image"], [args.image_size, args.image_size])

            anchor_classes, anchor_bboxes = tf.numpy_function(
                bboxes_utils.bboxes_training,  # name
                [anchors, example["classes"], example["bboxes"], args.iou_threshold],  # param values
                (tf.int64, tf.float32)  # return types
            )
            anchor_classes_one_hot = tf.one_hot(anchor_classes - 1, SVHN.LABELS)

            output = {
                "classes": tf.ensure_shape(anchor_classes_one_hot, [len(anchors), SVHN.LABELS]),
                "bboxes": tf.ensure_shape(anchor_bboxes, [len(anchors), 4])
            }

            sample_weights = {
                "classes": 1,
                "bboxes": tf.cast(anchor_classes > 0, tf.int32)
            }
            return resized_image, output, sample_weights

        if training:
            dataset = dataset.map(prepare_data)
            dataset = dataset.shuffle(buffer_size=10000, seed=args.seed)
        else:
            dataset = dataset.map(lambda example: (
            tf.image.resize(example["image"], [args.image_size, args.image_size]), tf.shape(example["image"])))
        dataset = dataset.batch(args.batch_size)

        return dataset

    train = create_dataset(svhn.train, True)
    dev = create_dataset(svhn.dev, False)
    test = create_dataset(svhn.test, False)

    # 3) Regularization structures
    if args.l2:
        regularizer = tf.keras.regularizers.L2(args.l2)
    else:
        regularizer = None

    def bn_relu(input):
        if args.batch_norm:
            return tf.keras.layers.ReLU()(tf.keras.layers.BatchNormalization()(input))
        else:
            return tf.keras.layers.ReLU()(input)

    if not args.decay or args.decay in ["None", "none"]:
        learning_rate = args.learning_rate
    else:
        decay_steps = (len(train) / args.batch_size) * args.epochs
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

    # 4) Load efficient net
    # change dynamic_input_shape in case of batching with size 1 and different sizes
    efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False, dynamic_input_shape=False)
    efficientnet_b0.trainable = args.fine_tuning

    # 5) create model
    class Model(tf.keras.Model):

        def __init__(self, args: argparse.Namespace) -> None:
            inputs = tf.keras.Input(shape=(args.image_size, args.image_size, 3))

            pyramid_output = efficientnet_b0(inputs)[len(efficientnet_b0.outputs) - args.level]
            eff_representation_size = int(args.image_size / 2 ** args.level)  # 14
            pyramid_output = tf.keras.layers.Dropout(args.dropout)(pyramid_output)

            # classification head
            classes_conv1 = bn_relu(
                tf.keras.layers.Conv2D(args.conv_filters, 3, 1, "same", kernel_regularizer=regularizer)(pyramid_output))
            classes_conv2 = bn_relu(
                tf.keras.layers.Conv2D(args.conv_filters, 3, 1, "same", kernel_regularizer=regularizer)(classes_conv1))
            classes_conv3 = bn_relu(
                tf.keras.layers.Conv2D(args.conv_filters, 3, 1, "same", kernel_regularizer=regularizer)(classes_conv2))
            classes_conv4 = tf.keras.layers.Conv2D(SVHN.LABELS, 3, 1, "same", activation=tf.nn.sigmoid,
                                                   kernel_regularizer=regularizer)(classes_conv3)
            classes_output_reshaped = tf.keras.layers.Reshape((eff_representation_size ** 2, SVHN.LABELS),
                                                              name="classes")(classes_conv4)

            # bbox regression head
            bbox_conv1 = bn_relu(
                tf.keras.layers.Conv2D(args.conv_filters, 3, 1, "same", kernel_regularizer=regularizer)(pyramid_output))
            bbox_conv2 = bn_relu(
                tf.keras.layers.Conv2D(args.conv_filters, 3, 1, "same", kernel_regularizer=regularizer)(bbox_conv1))
            bbox_conv3 = bn_relu(
                tf.keras.layers.Conv2D(args.conv_filters, 3, 1, "same", kernel_regularizer=regularizer)(bbox_conv2))
            bbox_conv4 = tf.keras.layers.Conv2D(4, 3, 1, "same", kernel_regularizer=regularizer)(bbox_conv3)
            bbox_output_reshaped = tf.keras.layers.Reshape((eff_representation_size ** 2, 4), name="bboxes")(bbox_conv4)

            outputs = {
                "classes": classes_output_reshaped,
                "bboxes": bbox_output_reshaped
            }

            super().__init__(inputs=inputs, outputs=outputs)

            self.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss={  # keys fit to output dict
                    "classes": tf.keras.losses.BinaryFocalCrossentropy(),
                    "bboxes": tf.keras.losses.Huber()
                },
                metrics={
                    "classes": [],
                    "bboxes": []
                }  # better call dataset.evaluate instead -> implement as callback (example: example_keras_models in 03)
            )

        @staticmethod
        def bboxes_from_fast_rcnn_batch(anchors, fast_rcnn_batch):
            batch_elements = tf.unstack(fast_rcnn_batch)
            processed = []
            for element in batch_elements:
                result = bboxes_utils.bboxes_from_fast_rcnn(anchors, element)
                processed.append(result)
            output = tf.stack(processed)
            return output

        # Override `predict_step` to perform non-max suppression and rescaling of bounding boxes
        def predict_step(self, data):
            # tf.print("enter predict step")
            images, sizes = data

            # predict
            y_pred = self(images, training=False)
            classes, bboxes = y_pred["classes"], y_pred["bboxes"]

            # transform bboxes after NN back to normal representation
            # tf.print("bboxes shape", tf.shape(bboxes))
            bboxes = tf.numpy_function(
                self.bboxes_from_fast_rcnn_batch,  # name
                [anchors, bboxes],  # param values
                (tf.float32)  # return types
            )
            # tf.print("bboxes shape after transform", tf.shape(bboxes))
            # tf.print("classes shape", tf.shape(classes))

            # non max suppression
            bboxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                bboxes[:, :, tf.newaxis], classes, 5, 5, args.iou_prediction, score_threshold=args.score_threshold)
            # tf.print("---------")
            # tf.print("valid_detections.shape", tf.shape(valid_detections))
            # tf.print(valid_detections)
            # tf.print(tf.unique(valid_detections))
            # tf.print("---------")
            # tf.print("bboxes.shape", tf.shape(bboxes))
            # tf.print("bboxes", bboxes)
            # tf.print("---------")

            # resize bboxes to original size
            bboxes *= tf.cast(sizes[:, 0], tf.float32)[:, tf.newaxis, tf.newaxis]

            return classes, bboxes, valid_detections

    model = Model(args)
    # model.summary()

    # 6) train model
    def create_predictions(model, data, filename):  # used for dev and test evaluation
        # Generate test set annotations, but in `args.logdir` to allow parallel execution.
        os.makedirs(args.logdir, exist_ok=True)
        with open(os.path.join(args.logdir, filename), "w", encoding="utf-8") as predictions_file:
            # : Predict the digits and their bounding boxes on the test set.
            # Assume that for a single test image we get
            # - `predicted_classes`: a 1D array with the predicted digits,
            # - `predicted_bboxes`: a [len(predicted_classes), 4] array with bboxes;
            predictions = model.predict(data)
            predicted_classes, predicted_bboxes, valid_detections = predictions

            for test_image_idx in range(predicted_classes.shape[0]):
                output = []
                # limit outputs to valid outputs from non max suppression
                for valid_idx in range(valid_detections[test_image_idx]):
                    label = int(predicted_classes[test_image_idx][valid_idx])
                    bbox = predicted_bboxes[test_image_idx][valid_idx]

                    output += [label] + list(bbox)
                print(*output, file=predictions_file)

    def evaluate_dev(epoch, logs):
        filename = "svhn_dev.txt"
        # create predictions in file
        create_predictions(model, dev, filename)
        # read file and evaluate it
        with open(os.path.join(args.logdir, filename), "r", encoding="utf-8-sig") as predictions_file:
            accuracy = SVHN.evaluate_file(svhn.dev, predictions_file)
            logs.update({"val_accuracy": accuracy})

    best_checkpoint_path = os.path.join(args.logdir, "svhn_competition.ckpt")
    model.fit(
        train.take(1), batch_size=args.batch_size, epochs=args.epochs,
        callbacks=[tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0),
                   tf.keras.callbacks.LambdaCallback(on_epoch_end=evaluate_dev),
                   tf.keras.callbacks.ModelCheckpoint(filepath=best_checkpoint_path,
                                                      save_weights_only=False, monitor='val_accuracy',
                                                      mode='max', save_best_only=True)]
    )

    # 7) predict test set with best model stored in checkpoint
    best_model = tf.keras.models.load_model(best_checkpoint_path)
    create_predictions(best_model, test, "svhn_competition.txt")


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
