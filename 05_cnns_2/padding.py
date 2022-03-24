import numpy as np
import tensorflow as tf
# simple two dim
def pad_2d(tensor):
    row_concats = []
    for row in tensor:
        row_concat = None
        for column in row:
            # pad single elements
            padded = tf.pad(tf.reshape(column, [-1, 1]), tf.constant([[0,1], [0,1]]))
            # padded = tf.pad(column, tf.constant([[0,1], [0,1]]))
            if row_concat is None:
                row_concat = padded
            else:
                # concat single paddings "horizontally"
                row_concat = tf.concat([row_concat, padded], 1)
        row_concats.append(row_concat)
    # concat padded rows "vertically"
    return tf.concat(row_concats, 0)

# 4 dim with batches, channels
# TODO: improve performance!!
def pad_batch_of_images(batch):
    images_padded = []
    for image in batch:
        image = tf.image.convert_image_dtype(image, tf.float32)
        channels_padded = []
        for channel in range(image.shape[2]):
            img_channel = image[:,:,channel]
            # pad "between" pixels
            channel_padded = pad_2d(img_channel)
            # add surrounding padding (TODO: now added left and top as well, necessary?)
            # channel_padded = tf.pad(channel_padded, tf.constant([[1,1], [1,1]]))
            channels_padded.append(channel_padded)
        image_padded = tf.stack(channels_padded, 2)
        images_padded.append(image_padded)
        # print("image padded shape", image_padded.shape)
    batch_padded = tf.stack(images_padded)
    return batch_padded


if __name__ == '__main__':
    print(pad_2d(np.array([[1, 2, 3], [4, 5, 6]])))
    c = tf.random.uniform([10, 13, 13, 5])

    c_padded = pad_batch_of_images(c)
    print("final shape", c_padded.shape)
