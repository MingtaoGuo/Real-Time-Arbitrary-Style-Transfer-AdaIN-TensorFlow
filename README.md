Introduction
--------------

This code mainly implement the paper [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868) which address the problem of arbitrary style transfer in real-time. The main contribution of this paper is 'Adaptive Instance Normalization(AdaIN)' proposed by Xun Huang etc. 

![](https://github.com/MingtaoGuo/Real-time-Arbitrary-Style-Transfer/blob/master/Figures/AdaIN.jpg)

Procedure of this method is as shown in follow figure.

![](https://github.com/MingtaoGuo/Real-time-Arbitrary-Style-Transfer/blob/master/Figures/style_transfer_network.jpg)

How to train the network
------------------------

Firstly, you should download the data sets ([MSCOCO]() and [wikiart]()). Putting them into the file 'content' and 'style' respectively.
