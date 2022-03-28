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
logger = logging.getLogger('Segmentation')

import numpy as np
import tensorflow as tf

from cags_dataset import CAGS
import efficient_net

# : Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=94, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--logging_level", default="warning", type=str, help="Logging level")
parser.add_argument("--fine_tuning", default=True, type=bool, help="Optionally fine tune the efficient net core.")
parser.add_argument("--batch_norm", default=True, type=bool, help="Batch normalization of conv. layers.")
parser.add_argument("--l2", default=0.00, type=float, help="L2 regularization.")
parser.add_argument("--decay", default="linear", type=str, help="Learning decay rate type")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default=0.00001, type=float, help="Final learning rate.")

# Params used for best model:
# cags_segmentation.py-2022-03-19_174704-bn=True,bs=50,d=None,e=94,ft=True,l=0.0,lr=0.001,lrf=0.00001,ll=warning,s=42,t=1


def main(args: argparse.Namespace) -> None:
    print(args)
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
    cags = CAGS()

    img_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range = 30,
        horizontal_flip = True,
	vertical_flip = True,
        fill_mode = 'nearest'
    )

    def prepare_dataset(dataset, training):
        def concatination(element):
            return tf.concat([element["image"],element["mask"]],axis=-1)
        dataset = dataset.map(concatination)
        def augment(element):
            return tf.ensure_shape(tf.numpy_function(img_generator.random_transform, [element], tf.float32),(224,224,4))
        dataset = dataset.map(augment)
        def slicing(element):
            mask = tf.slice(element, begin = [0,0,3], size = [224,224,1])
            image = tf.slice(element, begin = [0,0,0], size = [224,224,3])
            return image, mask
        dataset = dataset.map(slicing)
        if training:
            dataset = dataset.shuffle(buffer_size = 1500, seed = args.seed)
        dataset = dataset.batch(args.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def prepare_else(dataset, training):
        def divide(element):
            return element["image"], element["mask"]
        dataset = dataset.map(divide)
        if training:
            dataset = dataset.shuffle(buffer_size = 1500, seed = args.seed)
        dataset = dataset.batch(args.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    train = prepare_dataset(cags.train, True)
    """ import matplotlib.pyplot as plt """
    """ for images, masks in train:
        plt.imshow(np.concatenate([images[0], np.repeat(masks[0], 3, axis=2)], axis=1))
        plt.show() """
    dev = prepare_else(cags.dev, False)
    test = prepare_else(cags.test, False)
    # Load the EfficientNet-B0 model
    efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False)
    # efficientnet_b0.summary()
    logger.debug("Efficient Net Outputs:")
    for x in efficientnet_b0.output:
        logger.debug(str(x))
    efficientnet_b0.trainable = args.fine_tuning
    logger.debug("-" * 100)

    # : Create the model and train it

    if args.l2:
        regularizer = tf.keras.regularizers.L2(args.l2)
    else:
        regularizer = None

    def bn_relu(input):
        if args.batch_norm:
            return tf.keras.layers.ReLU()(tf.keras.layers.BatchNormalization()(input))
        else:
            return input

    if not args.decay == "None":
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
            learning_rate = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate = args.learning_rate,
                                                                    decay_steps = decay_steps,
                                                                    alpha = args.learning_rate_final)
        else:
            raise NotImplementedError("Use only 'linear' or 'exponential' as LR scheduler")

    # Input layer
    logger.debug(' Input layer '.center(40, '*'))
    inputs = tf.keras.Input(shape=(CAGS.H, CAGS.W, CAGS.C))
    logger.debug("inputs " + str(inputs))
    # inputs_conv1 = bn_relu(tf.keras.layers.Conv2D(112, 3, 1, "same")(
    #     inputs))
    # inputs_conv2 = bn_relu(tf.keras.layers.Conv2D(112, 3, 1, "same")(
    #     inputs_conv1))
    # print("inputs_conv2", inputs_conv2)
    hidden = efficientnet_b0(inputs)

    # L1
    logger.debug(' L1 '.center(40, '*'))
    deepest_resolution_layer_7_7_1280 = hidden[1]
    logger.debug("deepest_resolution_layer_7_7_1280 " + str(deepest_resolution_layer_7_7_1280))
    conv1_7_7_1280 = bn_relu(tf.keras.layers.Conv2D(1280, 3, 1, "same", kernel_regularizer=regularizer)(
        deepest_resolution_layer_7_7_1280))
    logger.debug("conv1_7_7_1280 " + str(conv1_7_7_1280))
    conv2_7_7_1280 = bn_relu(tf.keras.layers.Conv2D(1280, 3, 1, "same", kernel_regularizer=regularizer)(conv1_7_7_1280))
    logger.debug("conv2_7_7_1280 " + str(conv2_7_7_1280))
    trans_14_14_112 = bn_relu(
        tf.keras.layers.Conv2DTranspose(112, 3, 2, "same", kernel_regularizer=regularizer)(conv2_7_7_1280))
    logger.debug("trans_14_14_112 " + str(trans_14_14_112))

    # L2
    logger.debug(' L2 '.center(40, '*'))
    concat_14_14_224 = tf.keras.layers.Concatenate()([hidden[2], trans_14_14_112])
    logger.debug("concat_14_14_224 " + str(concat_14_14_224))
    conv1_14_14_112 = bn_relu(
        tf.keras.layers.Conv2D(112, 3, 1, "same", kernel_regularizer=regularizer)(concat_14_14_224))
    logger.debug("conv1_14_14_112 " + str(conv1_14_14_112))
    conv2_14_14_112 = bn_relu(
        tf.keras.layers.Conv2D(112, 3, 1, "same", kernel_regularizer=regularizer)(conv1_14_14_112))
    logger.debug("conv2_14_14_112 " + str(conv2_14_14_112))
    trans_28_28_40 = bn_relu(
        tf.keras.layers.Conv2DTranspose(40, 3, 2, "same", kernel_regularizer=regularizer)(conv2_14_14_112))
    logger.debug("trans_28_28_40 " + str(trans_28_28_40))

    # L3
    logger.debug(' L3 '.center(40, '*'))
    concat_28_28_80 = tf.keras.layers.Concatenate()([hidden[3], trans_28_28_40])
    logger.debug("concat_28_28_80 " + str(concat_28_28_80))
    conv1_28_28_40 = bn_relu(tf.keras.layers.Conv2D(40, 3, 1, "same", kernel_regularizer=regularizer)(concat_28_28_80))
    logger.debug("conv1_28_28_40 " + str(conv1_28_28_40))
    conv2_28_28_40 = bn_relu(tf.keras.layers.Conv2D(40, 3, 1, "same", kernel_regularizer=regularizer)(conv1_28_28_40))
    logger.debug("conv2_28_28_40 " + str(conv2_28_28_40))
    trans_56_56_24 = bn_relu(tf.keras.layers.Conv2DTranspose(24, 3, 2, "same", kernel_regularizer=regularizer)(conv2_28_28_40))
    logger.debug("trans_56_56_24 " + str(trans_56_56_24))

    # L4
    logger.debug(' L4 '.center(40, '*'))
    concat_56_56_48 = tf.keras.layers.Concatenate()([hidden[4], trans_56_56_24])
    logger.debug("concat_56_56_48 " + str(concat_56_56_48))
    conv1_56_56_24 = bn_relu(tf.keras.layers.Conv2D(24, 3, 1, "same", kernel_regularizer=regularizer)(concat_56_56_48))
    logger.debug("conv1_56_56_24 " + str(conv1_56_56_24))
    conv2_56_56_24 = bn_relu(tf.keras.layers.Conv2D(24, 3, 1, "same", kernel_regularizer=regularizer)(conv1_56_56_24))
    logger.debug("conv2_56_56_24 " + str(conv2_56_56_24))
    trans_112_112_16 = bn_relu(tf.keras.layers.Conv2DTranspose(16, 3, 2, "same", kernel_regularizer=regularizer)(conv2_56_56_24))
    logger.debug("trans_112_112_16 " + str(trans_112_112_16))

    # L5
    logger.debug(' L5 '.center(40, '*'))
    concat_112_112_32 = tf.keras.layers.Concatenate()([hidden[5], trans_112_112_16])
    logger.debug("concat_112_112_32 " + str(concat_112_112_32))
    conv1_112_112_16 = bn_relu(tf.keras.layers.Conv2D(16, 3, 1, "same", kernel_regularizer=regularizer)(concat_112_112_32))
    logger.debug("conv1_112_112_16 " + str(conv1_112_112_16))
    conv2_112_112_3 = bn_relu(tf.keras.layers.Conv2D(8, 3, 1, "same", kernel_regularizer=regularizer)(conv1_112_112_16))
    logger.debug("conv2_112_112_3 " + str(conv2_112_112_3))
    trans_224_224_3 = bn_relu(tf.keras.layers.Conv2DTranspose(3, 3, 2, "same", kernel_regularizer=regularizer)(conv2_112_112_3))
    logger.debug("trans_224_224_3 " + str(trans_224_224_3))

    # Output layer
    logger.debug(' Output Layer '.center(40, '*'))
    concat_224_224_6 = tf.keras.layers.Concatenate()([inputs, trans_224_224_3])
    logger.debug("concat_224_224_6 " + str(concat_224_224_6))
    conv1_224_224_6 = bn_relu(tf.keras.layers.Conv2D(6, 3, 1, "same", kernel_regularizer=regularizer)(concat_224_224_6))
    logger.debug("conv1_224_224_6 " + str(conv1_224_224_6))
    conv2_224_224_6 = bn_relu(tf.keras.layers.Conv2D(6, 3, 1, "same", kernel_regularizer=regularizer)(conv1_224_224_6))
    logger.debug("conv2_224_224_6 " + str(conv2_224_224_6))
    outputs = tf.keras.layers.Conv2D(1, 1, 1, "same", activation='sigmoid')(conv2_224_224_6)
    logger.debug("outputs " + str(outputs))

    # compose and train model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[CAGS.MaskIoUMetric(), 'accuracy']
    )
    # model.summary()
    model.fit(train, batch_size=args.batch_size, epochs=args.epochs, validation_data=dev,
              callbacks=[
                  tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)], verbose = 2
              )
    # exit()
    print(args)

    model.save(os.path.join(args.logdir, "cags_segmentation.h5"), include_optimizer=True)
    
    #model = tf.keras.models.load_model(os.path.join(args.logdir, "cags_segmentation.h5"))
    # use best checkpoint to make predictions

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_segmentation.txt"), "w", encoding="utf-8") as predictions_file:
        # : Predict the masks on the test set
        test_masks = model.predict(test)

        for mask in test_masks:
            zeros, ones, runs = 0, 0, []
            for pixel in np.reshape(mask >= 0.5, [-1]):
                if pixel:
                    if zeros or (not zeros and not ones):
                        runs.append(zeros)
                        zeros = 0
                    ones += 1
                else:
                    if ones:
                        runs.append(ones)
                        ones = 0
                    zeros += 1
            runs.append(zeros + ones)
            print(*runs, file=predictions_file)


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

