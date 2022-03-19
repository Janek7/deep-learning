#!/usr/bin/env python3
import argparse
import datetime
import os
import re

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from cags_dataset import CAGS
import efficient_net

# TODO: Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--fine_tuning", default=True, type=bool, help="Optionally fine tune the efficient net core.")
parser.add_argument("--batch_norm", default=False, type=bool, help="Batch normalization of conv. layers.")


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

    def prepare_dataset(dataset, training):
        def create_inputs(element):
            return element["image"], element["mask"]

        dataset = dataset.map(create_inputs)
        if training:
            dataset = dataset.shuffle(len(dataset))
        dataset = dataset.batch(1)
        dataset = dataset.take(1)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    train = prepare_dataset(cags.train, True)
    # for batch in train:
    #     for image, mask in zip(batch[0], batch[1]):
    #         print(image.shape, mask.shape)
    #         break
    #     break
    dev = prepare_dataset(cags.dev, False)
    test = prepare_dataset(cags.test, False)

    # Load the EfficientNet-B0 model
    efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False)
    # efficientnet_b0.summary()
    for x in efficientnet_b0.output:
        print(x)
    efficientnet_b0.trainable = args.fine_tuning
    print("-" * 100)

    # : Create the model and train it

    def bn_relu(input):
        if args.batch_norm:
            return tf.keras.layers.ReLU()(tf.keras.layers.BatchNormalization()(input))
        else:
            return input

    # Input layer
    print(' Input layer '.center(40, '*'))
    inputs = tf.keras.Input(shape=(CAGS.H, CAGS.W, CAGS.C))
    # print("inputs", inputs)
    # inputs_conv1 = bn_relu(tf.keras.layers.Conv2D(112, 3, 1, "same")(
    #     inputs))
    # inputs_conv2 = bn_relu(tf.keras.layers.Conv2D(112, 3, 1, "same")(
    #     inputs_conv1))
    # print("inputs_conv2", inputs_conv2)
    hidden = efficientnet_b0(inputs)

    # L1
    print(' L1 '.center(40, '*'))
    layer_7_7_1280 = hidden[1]
    print("layer_7_7_1280", layer_7_7_1280)
    conv1_7_7_1280 = bn_relu(
        tf.keras.layers.Conv2D(1280, 3, 1, "same")(layer_7_7_1280))
    print("conv1_7_7_1280", conv1_7_7_1280)
    conv2_7_7_1280 = bn_relu(tf.keras.layers.Conv2D(1280, 3, 1, "same")(conv1_7_7_1280))
    print("conv2_7_7_1280", conv2_7_7_1280)
    trans_14_14_112 = bn_relu(tf.keras.layers.Conv2DTranspose(112, 3, 2, "same")(conv2_7_7_1280))
    print("trans_14_14_112", trans_14_14_112)

    # L2
    print(' L2 '.center(40, '*'))
    concat_14_14_224 = tf.keras.layers.Concatenate()([hidden[2], trans_14_14_112])
    print("concat_14_14_224", concat_14_14_224)
    conv1_14_14_112 = bn_relu(tf.keras.layers.Conv2D(112, 3, 1, "same")(concat_14_14_224))
    print("conv1_14_14_112", conv1_14_14_112)
    conv2_14_14_112 = bn_relu(tf.keras.layers.Conv2D(112, 3, 1, "same")(conv1_14_14_112))
    print("conv2_14_14_112", conv2_14_14_112)
    trans_28_28_40 = bn_relu(tf.keras.layers.Conv2DTranspose(40, 3, 2, "same")(conv2_14_14_112))
    print("trans_28_28_40", trans_28_28_40)

    # L3
    print(' L3 '.center(40, '*'))
    concat_28_28_80 = tf.keras.layers.Concatenate()([hidden[3], trans_28_28_40])
    print("concat_28_28_80", concat_28_28_80)
    conv1_28_28_40 = bn_relu(tf.keras.layers.Conv2D(40, 3, 1, "same")(concat_28_28_80))
    print("conv1_28_28_40", conv1_28_28_40)
    conv2_28_28_40 = bn_relu(tf.keras.layers.Conv2D(40, 3, 1, "same")(conv1_28_28_40))
    print("conv2_28_28_40", conv2_28_28_40)
    trans_56_56_24 = bn_relu(tf.keras.layers.Conv2DTranspose(24, 3, 2, "same")(conv2_28_28_40))
    print("trans_56_56_24", trans_56_56_24)

    # L4
    print(' L4 '.center(40, '*'))
    concat_56_56_48 = tf.keras.layers.Concatenate()([hidden[4], trans_56_56_24])
    print("concat_56_56_48", concat_56_56_48)
    conv1_56_56_24 = bn_relu(tf.keras.layers.Conv2D(24, 3, 1, "same")(concat_56_56_48))
    print("conv1_56_56_24", conv1_56_56_24)
    conv2_56_56_24 = bn_relu(tf.keras.layers.Conv2D(24, 3, 1, "same")(conv1_56_56_24))
    print("conv2_56_56_24", conv2_56_56_24)
    trans_112_112_16 = bn_relu(tf.keras.layers.Conv2DTranspose(16, 3, 2, "same")(conv2_56_56_24))
    print("trans_112_112_16", trans_112_112_16)

    # L5
    print(' L5 '.center(40, '*'))
    concat_112_112_32 = tf.keras.layers.Concatenate()([hidden[5], trans_112_112_16])
    print("concat_112_112_32", concat_112_112_32)
    conv1_112_112_16 = bn_relu(tf.keras.layers.Conv2D(16, 3, 1, "same")(concat_112_112_32))
    print("conv1_112_112_16", conv1_112_112_16)
    conv2_112_112_3 = bn_relu(tf.keras.layers.Conv2D(8, 3, 1, "same")(conv1_112_112_16))
    print("conv2_112_112_3", conv2_112_112_3)
    trans_224_224_3 = bn_relu(tf.keras.layers.Conv2DTranspose(3, 3, 2, "same")(conv2_112_112_3))
    print("trans_224_224_3", trans_224_224_3)

    # Output layer
    print(' Output Layer '.center(40, '*'))
    concat_224_224_6 = tf.keras.layers.Concatenate()([inputs, trans_224_224_3])
    print("concat_224_224_6", concat_224_224_6)
    conv1_224_224_6 = bn_relu(tf.keras.layers.Conv2D(6, 3, 1, "same")(concat_224_224_6))
    print("conv1_224_224_6", conv1_224_224_6)
    conv2_224_224_6 = bn_relu(tf.keras.layers.Conv2D(6, 3, 1, "same")(conv1_224_224_6))
    print("conv2_224_224_6", conv2_224_224_6)
    outputs = tf.keras.layers.Conv2D(1, 1, 1, "same", activation='sigmoid')(conv2_224_224_6)
    print("outputs", outputs)

    # compose and train model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[CAGS.MaskIoUMetric(), 'accuracy']
    )
    best_checkpoint_path = os.path.join(args.logdir, "cags_segmentation.ckpt")
    model.fit(train, batch_size=args.batch_size, epochs=args.epochs, validation_data=dev,
              callbacks=[tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0),
                         tf.keras.callbacks.ModelCheckpoint(filepath=best_checkpoint_path, save_weights_only=False,
                                                            monitor='val_accuracy', mode='max', save_best_only=True)]
              )
    # exit()
    print(args)

    # use best checkpoint to make predictions
    model = tf.keras.models.load_model(best_checkpoint_path)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_segmentation.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the masks on the test set
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
    main(args)
