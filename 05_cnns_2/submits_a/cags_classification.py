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

from cags_dataset import CAGS
import efficient_net

# : Define reasonable defaults and optionally more parameters
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--fine_tuning", default=True, type=bool, help="Optionally fine tune the efficient net core.")
parser.add_argument("--l2", default=0.01, type=float, help="L2 regularization.")
parser.add_argument("--label_smoothing", default=0.0, type=float, help="Label smoothing.")
parser.add_argument("--decay", default='linear', type=str, help="Learning decay rate type")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default=0.0001, type=float, help="Final learning rate.")

# params used for submitted model:
# --batch_size=50 --epochs=50 --seed=42 --threads=1 --fine_tuning=True --l2=0.01 --label_smoothing=0 --decay=linear --learning_rate=0.001 --learning_rate_final=0.0001

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
        horizontal_flip = True,
        fill_mode='nearest'
    )
    def prepare_dataset(dataset, training):
        def create_inputs(element):
            return tf.ensure_shape(tf.numpy_function(img_generator.random_transform, [element["image"]], tf.float32),(224,224,3)), element["label"]
        dataset = dataset.map(create_inputs)
        if training:
            dataset = dataset.shuffle(5000, seed = args.seed)
        dataset = dataset.batch(args.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    def prepare_other(dataset, training):
        def create_inputs(element):
            return element["image"], element["label"]
        dataset = dataset.map(create_inputs)
        if training:
            dataset = dataset.shuffle(5000, seed = args.seed)
        dataset = dataset.batch(args.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    train = prepare_dataset(cags.train, True)
    dev = prepare_other(cags.dev, False)
    test = prepare_other(cags.test, False)

    # Load the EfficientNet-B0 model
    efficientnet_b0 = efficient_net.pretrained_efficientnet_b0(include_top=False)
    efficientnet_b0.trainable = args.fine_tuning

    # : Create the model and train it
    if args.l2:
        regularizer = tf.keras.regularizers.L2(args.l2)
    else:
        regularizer = None

    inputs = tf.keras.Input(shape=(CAGS.H, CAGS.W, CAGS.C))
    hidden = efficientnet_b0(inputs)
    hidden2 = tf.keras.layers.Dense(1024, activation = 'relu')(hidden[0])
    hidden2 = tf.keras.layers.Dense(1024, activation = 'relu')(hidden2)
    outputs = tf.keras.layers.Dense(len(CAGS.LABELS), activation=tf.nn.softmax, kernel_regularizer=regularizer)(hidden2)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    # set learning rate decay for the model
    if not args.decay:
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
            # print(learning_rate(decay_steps))
        elif args.decay == "cosine":
            learning_rate = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate = args.learning_rate,
                                                                    decay_steps = decay_steps,
                                                                    alpha = args.learning_rate_final
            )
        else:
            raise NotImplementedError("Not a valid schedule for decay")

    # compile and train model, depending on the fact if we want to smoothe the output
    if not args.label_smoothing:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    else:
        loss = tf.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)
        metrics = [tf.metrics.CategoricalAccuracy(name="accuracy")]

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=loss, metrics=metrics)

    model.fit(
        train, batch_size=args.batch_size, epochs=args.epochs, validation_data=dev,
        callbacks=[tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)], verbose = 2
    )

    # use best checkpoint to make predictions
    model.save(os.path.join(args.logdir, "cags_classification.h5"), include_optimizer=True)
    model = tf.keras.models.load_model(os.path.join(args.logdir, "cags_classification.h5"))

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_classification.txt"), "w", encoding="utf-8") as predictions_file:
        # : Predict the probabilities on the test set
        test_probabilities = model.predict(test)

        for probs in test_probabilities:
            print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

