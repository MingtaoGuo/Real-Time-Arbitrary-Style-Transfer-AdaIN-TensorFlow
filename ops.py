import tensorflow as tf
import tensorflow.contrib as contrib

epsilon = 1e-8
def conv(name, inputs, nums_out, ksize, strides, padding="VALID"):
    inputs = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
    nums_in = int(inputs.shape[-1])
    W = tf.get_variable("W"+name, [ksize, ksize, nums_in, nums_out], initializer=contrib.layers.xavier_initializer(), trainable=True)
    b = tf.get_variable("b"+name, [nums_out], initializer=tf.constant_initializer(0.), trainable=True)
    return tf.nn.conv2d(inputs, W, [1, strides, strides, 1], padding) + b

def deconv(name, inputs, nums_out, ksize, strides, padding="SAME"):
    nums_in = int(inputs.shape[3])
    h = tf.shape(inputs)[1]
    w = tf.shape(inputs)[2]
    W = tf.get_variable("W" + name, [ksize, ksize, nums_in, nums_out], initializer=contrib.layers.xavier_initializer(), trainable=True)
    b = tf.get_variable("b" + name, [nums_out], initializer=tf.constant_initializer(0.), trainable=True)
    inputs = tf.image.resize_nearest_neighbor(inputs, [h*strides, w*strides])
    return tf.nn.conv2d(inputs, W, [1, 1, 1, 1], padding) + b

def mapping(x):
    max = tf.reduce_max(x)
    min = tf.reduce_min(x)
    return (x - min) * 255 / (max - min + 1e-10)


def AdaIn(x, y):
    mu_x, var_x = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
    mu_y, var_y = tf.nn.moments(y, axes=[1, 2], keep_dims=True)
    sigma_x, sigma_y = tf.sqrt(var_x + epsilon), tf.sqrt(var_y + epsilon)
    return sigma_y * (x - mu_x) / sigma_x + mu_y

def content_loss(feature_c, t):
    return tf.reduce_sum(tf.reduce_mean(tf.square(feature_c - t), [1, 2]))

def style_loss(feature_bank_g, feature_bank_s):
    layers = ["relu1_1", "relu2_1", "relu3_1", "relu4_1"]
    L_s = 0
    for layer in layers:
        mu_g, var_g = tf.nn.moments(feature_bank_g[layer], axes=[1, 2], keep_dims=True)
        mu_s, var_s = tf.nn.moments(feature_bank_s[layer], axes=[1, 2], keep_dims=True)
        sigma_g, sigma_s = tf.sqrt(var_g), tf.sqrt(var_s)
        L_s = L_s + tf.reduce_sum(tf.reduce_mean(tf.square(mu_g - mu_s), [1, 2])) + \
                    tf.reduce_sum(tf.reduce_mean(tf.square(sigma_g - sigma_s), [1, 2]))
    return L_s