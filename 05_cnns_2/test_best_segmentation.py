import numpy as np
import tensorflow as tf
from cags_dataset import CAGS


# Load the data
cags = CAGS()

def prepare_dataset(dataset, training):
    def create_inputs(element):
        return element["image"], element["mask"]

    dataset = dataset.map(create_inputs)
    if training:
        dataset = dataset.shuffle(len(dataset))
    dataset = dataset.batch(50)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

dev = prepare_dataset(cags.dev, False)
test = prepare_dataset(cags.test, False)

model = tf.keras.models.load_model("C:\\Users\\janek\\Development\\Git\\Prag\\deep-learning-lecture\\05_cnns_2\\logs-segmentation\\cags_segmentation.py-2022-03-19_174704-bn=True,bs=50,d=None,e=50,ft=True,l=0.0,lr=0.001,lrf=0.0001,ll=warning,s=42,t=1\\cags_segmentation.ckpt",
                          custom_objects={CAGS.MaskIoUMetric.__name__: CAGS.MaskIoUMetric})

e = model.evaluate(dev)
# print(e)
# 7/7 [==============================] - 31s 4s/step - loss: 0.1226 - iou: 0.8974 - accuracy: 0.9548
# [0.1226455345749855, 0.8973842263221741, 0.9547705054283142]

with open("cags_segmentation.txt", "w", encoding="utf-8") as predictions_file:
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