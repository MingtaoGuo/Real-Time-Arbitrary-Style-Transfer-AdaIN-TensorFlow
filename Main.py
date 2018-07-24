import tensorflow as tf
from PIL import Image
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from network import encoder, decoder
from ops import *
from utils import *



class main:
    def __init__(self):
        self.content = tf.placeholder("float", [None, None, None, 3])
        self.style = tf.placeholder("float", [None, None, None, 3])
        lamda = 2
        batch_size = 2
        en = encoder("encoder")
        de = decoder("decoder")
        feature_bank_c = en(preprocess(tf.reverse(self.content, [-1])))
        feature_bank_s = en(preprocess(tf.reverse(self.style, [-1])))
        t = AdaIn(feature_bank_c["relu4_1"], feature_bank_s["relu4_1"])
        self.styled_img = de(t)
        feature_bank_g = en(preprocess(tf.reverse(self.styled_img, [-1])))
        self.content_loss = content_loss(feature_bank_g["relu4_1"], t)
        self.style_loss = style_loss(feature_bank_g, feature_bank_s)
        self.Loss = self.content_loss + lamda * self.style_loss
        self.Opt = tf.train.AdamOptimizer(1e-4).minimize(self.Loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        content_list = os.listdir("./content/")
        style_list = os.listdir("./style/")
        saver = tf.train.Saver()
        for i in range(80000):
            content_batch_locs = np.random.randint(0, content_list.__len__(), batch_size)
            style_batch_locs = np.random.randint(0, style_list.__len__(), batch_size)
            content_batch = np.zeros([batch_size, 128, 128, 3])
            style_batch = np.zeros([batch_size, 128, 128, 3])
            for j in range(batch_size):
                try:
                    content_batch[j, :, :, :] = resize_and_crop(np.array(Image.open("./content/" + content_list[content_batch_locs[j]])), 128)
                    style_batch[j, :, :, :] = resize_and_crop(np.array(Image.open("./style/" + style_list[style_batch_locs[j]])), 128)
                except:
                    content_batch[j, :, :, :] = resize_and_crop(np.array(Image.open("./content/" + content_list[0])), 128)
                    style_batch[j, :, :, :] = resize_and_crop( np.array(Image.open("./style/" + style_list[0])), 128)

            self.sess.run(self.Opt, feed_dict={self.content: content_batch, self.style: style_batch})
            if i % 10 == 0:
                [styled_img, Loss, c_loss, s_loss] = self.sess.run([self.styled_img, self.Loss, self.content_loss, self.style_loss], feed_dict={self.content: content_batch, self.style: style_batch})
                print("Step: %d, Total_loss: %f, Content_loss: %f, Style_loss: %f"%(i, Loss, c_loss, s_loss))
                Image.fromarray(np.uint8(mapping(styled_img[0, :, :, :]))).save("./result/"+str(i)+".jpg")
            if i % 2000 == 0:
                saver.save(self.sess, "./trained_para_justin/model.ckpt")

def test(content_path, style_path, alpha=1.0):
    c = np.array(Image.open(content_path))
    c = np.reshape(c, [1, c.shape[0], c.shape[1], c.shape[2]])
    s = np.array(Image.open(style_path))
    s = np.reshape(s, [1, s.shape[0], s.shape[1], s.shape[2]])
    content = tf.placeholder("float", [1, c.shape[1], c.shape[2], 3])
    style = tf.placeholder("float", [1, s.shape[1], s.shape[2], 3])
    en = encoder("encoder")
    de = decoder("decoder")
    feature_bank_c = en(preprocess(tf.reverse(content, [-1])))["relu4_1"]
    feature_bank_s = en(preprocess(tf.reverse(style, [-1])))["relu4_1"]
    t = AdaIn(feature_bank_c, feature_bank_s)
    styled_img = de((1 - alpha) * feature_bank_c + alpha * t)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "./trained_para/model.ckpt")
    [styled_img] = sess.run([styled_img],feed_dict={content: c, style: s})
    Image.fromarray(np.uint8(styled_img[0, :, :, :])).show()


if __name__ == "__main__":
    istraining = True
    if istraining:
        main()
    else:
        test()
