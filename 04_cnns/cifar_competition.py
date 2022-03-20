#!/usr/bin/env python3

#CONTRIBUTORS
# Antonio Krizmanic - 2b193238-8e3c-11ec-986f-f39926f24a9c
# Janek Putz - e31a3cae-8e6c-11ec-986f-f39926f24a9c

#All of the values used in the creation of the model are set as default in the .add_argument() methods or stated afterwards.
#The .txt file was created in a seperate .py file locally
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") 
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from cifar10 import CIFAR10
parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=142, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--learning", default=0.001, type=float)
parser.add_argument("--location", default = "cifar10_competition.h5", type = str)

class Model(tf.keras.Model):
    def __init__(self, struct: str, args: argparse.Namespace) -> None:
        input = tf.keras.layers.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C])
        hidden = self.createTheModel(struct, input)
        output = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(hidden)
        super().__init__(inputs=input, outputs=output)
        
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1)
    
    def conv_layer(self, shape: str, previous_layer):
        filt = int(shape.split("-")[1])
        ks = int(shape.split("-")[2])
        stride = int(shape.split("-")[3])
        pad = shape.split("-")[4]
        return tf.keras.layers.Conv2D(filters = filt, kernel_size = ks,  strides=(stride,stride), padding = pad, activation=tf.nn.relu)(previous_layer)

    def clear_BN(self, previous_layer):
        return tf.keras.layers.BatchNormalization()(previous_layer)

    def batch_normalisation(self, shape: str, previous_layer):
        filt = int(shape.split("-")[1])
        ks = int(shape.split("-")[2])
        stride = int(shape.split("-")[3])
        pad = shape.split("-")[4]
        tempBN = tf.keras.layers.Conv2D(filters = filt, kernel_size = ks,  strides=(stride,stride), padding = pad, use_bias = None, activation=None)(previous_layer)
        tempBN = tf.keras.layers.BatchNormalization()(tempBN)
        return tf.keras.layers.ReLU()(tempBN)

    def max_pool(self, shape: str, previous_layer):
        pools = int(shape.split("-")[1])
        stride = int(shape.split("-")[2])
        return tf.keras.layers.MaxPool2D(pool_size= (pools,pools), strides = (stride,stride))(previous_layer)

    def avg_pool(self, shape: str, previous_layer):
        pools = int(shape.split("-")[1])
        stride = int(shape.split("-")[2])
        return tf.keras.layers.AveragePooling2D(pool_size= (pools,pools), strides = (stride,stride))(previous_layer)

    def flat(self, previous_layer):
        return tf.keras.layers.Flatten()(previous_layer)

    def drop(self, shape: str, previous_layer):
        dropout_rate = float(shape.split("-")[1])
        return tf.keras.layers.Dropout(rate = dropout_rate)(previous_layer)

    def hide(self, shape: str, previous_layer):
        hls = int(shape.split("-")[1])
        return tf.keras.layers.Dense(units = hls, activation = tf.nn.relu)(previous_layer) 

    def spatial(self, shape: str, previous_layer):
        rate_drop = float(shape.split("-")[1])
        return tf.keras.layers.SpatialDropout2D(rate=rate_drop)(previous_layer) 

    def residual_layer(self, shape: str, previous_layer):
        preserve = previous_layer
        tempRL = previous_layer
        print(f'{shape}, buraz')
        sequence_of_actions = shape.replace("R-[", "").replace("]", "").split(",")
        print(sequence_of_actions)
        for action in sequence_of_actions:
            if action.split("-")[0] == "C":
                tempRL = self.conv_layer(action,tempRL)
            elif action.split("-")[0] == "CB":
                tempRL = self.batch_normalisation(action,tempRL)
            else: 
                print("nema smisla staviti taj sloj u reg")
        return tf.keras.layers.Add()([preserve, tempRL])

    def ResNetBottleneck(self, shape: str, previous_layer):
        size = int(shape.split("-")[1])
        preserve = previous_layer
        tempRNB = previous_layer
        tempRNB = tf.keras.layers.Conv2D(filters = size/2, kernel_size = 1,  strides=1, padding = "same", activation=None, use_bias = None)(tempRNB)
        tempRNB = tf.keras.layers.BatchNormalization()(tempRNB)
        tempRNB = tf.keras.layers.ReLU()(tempRNB)
        tempRNB = tf.keras.layers.Conv2D(filters = size/2, kernel_size = 3,  strides=1, padding = "same", activation=None, use_bias = None)(tempRNB)
        tempRNB = tf.keras.layers.BatchNormalization()(tempRNB)
        tempRNB = tf.keras.layers.ReLU()(tempRNB)
        tempRNB = tf.keras.layers.Conv2D(filters = size, kernel_size = 1,  strides=1, padding = "same", activation=None, use_bias = None)(tempRNB)
        tempRNB = tf.keras.layers.BatchNormalization()(tempRNB)
        return tf.keras.layers.ReLU()(tf.keras.layers.Add()([preserve, tempRNB]))

    def createTheModel(self, structure: str, input_layer):
        hidden = input_layer
        model_list = structure.split(",")
        for layer in model_list:
            if layer.split("-")[0] == "C":
                hidden = self.conv_layer(layer,hidden)
            elif layer.split("-")[0] == "CB" and layer[-1] != "]":
                hidden = self.batch_normalisation(layer,hidden)
            elif layer.split("-")[0] == "M":
                hidden = self.max_pool(layer,hidden)
            elif layer.split("-")[0] == "R":
                strr = layer
                for other_half in model_list:
                    if other_half.split("-")[0] == "CB" and other_half[-1] == "]":
                        strr = strr +","+ other_half
                hidden = self.residual_layer(strr,hidden)
            elif layer.split("-")[0] == "F":
                hidden = self.flat(hidden)
            elif layer.split("-")[0] == "D":
                hidden = self.drop(layer,hidden)
            elif layer.split("-")[0] == "H":
                hidden = self.hide(layer,hidden)
            elif layer.split("-")[0] == "SD":
                hidden = self.spatial(layer,hidden)
            elif layer.split("-")[0] == "A":
                hidden = self.avg_pool(layer,hidden)
            elif layer.split("-")[0] == "RNB":
                hidden = self.ResNetBottleneck(layer,hidden)
            elif layer.split("-")[0] == "CBN":
                hidden = self.clear_BN(hidden)
            else:  
                continue
        return hidden

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

    # Load data
    cifar = CIFAR10()

    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range = 15,
        zoom_range = 0.15,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        horizontal_flip = True
    )

    model = Model("C-32-3-1-same,CBN,RNB-32,M-2-2,SD-0.15,C-64-3-1-same,CBN,RNB-64,M-2-2,SD-0.15,C-128-3-1-same,CBN,RNB-128,M-2-2,C-256-3-1-same,CBN,RNB-256,SD-0.15,F,H-64,H-64", args)
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate = args.learning,momentum = args.momentum),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy("accuracy")]
        )
    history = model.fit(
        train_generator.flow(x = cifar.train.data["images"], y=cifar.train.data["labels"], batch_size=args.batch_size, seed=args.seed),
        shuffle=False, epochs=args.epochs,
        validation_data=(cifar.dev.data["images"], cifar.dev.data["labels"]),
        callbacks=[tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1)],
        verbose = 2
    )

    model.save(os.path.join(args.logdir, args.location), include_optimizer=True)
    
    model = tf.keras.models.load_model('models\cifar10_competition.h5')
    predictions = model.predict(cifar.test.data['images'])
    predictions = tf.argmax(predictions, axis = -1)
    
    # Generate test set annotations, but in args.logdir to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open("cifar10_test.txt", "w", encoding="utf-8") as predictions_file:
        data = tf.data.Dataset.from_tensor_slices(predictions)
        for i in data:
                predictions_file.write(str(np.asarray(i)) + "\n")
    predictions_file.close()

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)


