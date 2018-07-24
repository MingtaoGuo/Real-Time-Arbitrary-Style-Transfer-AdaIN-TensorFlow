import scipy.io as sio
from ops import *


class decoder:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            inputs = tf.nn.relu(deconv("3_4", inputs, 256, 3, 2))
            inputs = tf.nn.relu(conv("3_3", inputs, 256, 3, 1))
            inputs = tf.nn.relu(conv("3_2", inputs, 256, 3, 1))
            inputs = tf.nn.relu(conv("3_1", inputs, 256, 3, 1))
            inputs = tf.nn.relu(deconv("2_2", inputs, 128, 3, 2))
            inputs = tf.nn.relu(conv("2_1", inputs, 128, 3, 1))
            inputs = tf.nn.relu(deconv("1_2", inputs, 64, 3, 2))
            inputs = tf.nn.relu(conv("1_1", inputs, 64, 3, 1))
            return tf.clip_by_value(conv("output", inputs, 3, 3, 1), 0, 255)

class encoder:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs):
        vgg_para = sio.loadmat("./vgg_para//vgg.mat")
        layers = vgg_para["layers"]
        feature_bank = {}
        with tf.variable_scope(self.name):
            for i in range(37):
                if layers[0, i][0, 0]["type"] == "conv":
                    w = layers[0, i][0, 0]["weights"][0, 0]
                    b = layers[0, i][0, 0]["weights"][0, 1]
                    with tf.variable_scope(str(i)):
                        inputs = tf.nn.conv2d(inputs, w, [1, 1, 1, 1], "SAME") + b
                elif layers[0, i][0, 0]["type"] == "relu":
                    inputs = tf.nn.relu(inputs)
                    feature_bank[layers[0, i][0, 0]["name"][0]] = inputs
                    if layers[0, i][0, 0]["name"][0] == "relu4_1":
                        return feature_bank
                else:
                    inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

