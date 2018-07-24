Introduction
--------------

This code mainly implement the paper [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868) which address the problem of arbitrary style transfer in real-time. The main contribution of this paper is 'Adaptive Instance Normalization(AdaIN)' proposed by Xun Huang etc. 

![](https://github.com/MingtaoGuo/Real-time-Arbitrary-Style-Transfer/blob/master/Figures/AdaIN.jpg)

Procedure of this method is as shown in follow figure.

![](https://github.com/MingtaoGuo/Real-time-Arbitrary-Style-Transfer/blob/master/Figures/style_transfer_network.jpg)

How to train the network
------------------------

Python packages you need:

1. python 3.x
2. tensorflow 1.4.0
3. numpy
4. scipy
5. pillow

Data sets you need:

1. Content images data sets ([MSCOCO]())
2. Style images data sets ([wikiart]())

Results of our code
--------------------

Style | ![](https://github.com/MingtaoGuo/Real-time-Arbitrary-Style-Transfer/blob/master/Figures/ori.jpg) | ![](https://github.com/MingtaoGuo/Real-time-Arbitrary-Style-Transfer/blob/master/Figures/ori1.jpg)
:-- | :--: | --:
![](https://github.com/MingtaoGuo/Real-time-Arbitrary-Style-Transfer/blob/master/Figures/s1.jpg) | ![](https://github.com/MingtaoGuo/Real-time-Arbitrary-Style-Transfer/blob/master/Figures/result1.jpg)  | ![](https://github.com/MingtaoGuo/Real-time-Arbitrary-Style-Transfer/blob/master/Figures/result3.jpg)
![](https://github.com/MingtaoGuo/Real-time-Arbitrary-Style-Transfer/blob/master/Figures/s2_3.jpg) | ![](https://github.com/MingtaoGuo/Real-time-Arbitrary-Style-Transfer/blob/master/Figures/result6.jpg)  | ![](https://github.com/MingtaoGuo/Real-time-Arbitrary-Style-Transfer/blob/master/Figures/result5.jpg)
![](https://github.com/MingtaoGuo/Real-time-Arbitrary-Style-Transfer/blob/master/Figures/fire.jpg) | ![](https://github.com/MingtaoGuo/Real-time-Arbitrary-Style-Transfer/blob/master/Figures/result7.jpg)  | ![](https://github.com/MingtaoGuo/Real-time-Arbitrary-Style-Transfer/blob/master/Figures/result4.jpg)
