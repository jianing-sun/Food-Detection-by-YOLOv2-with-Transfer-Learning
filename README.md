# Food Detection System Design by Deep Convolutional Neural Networks

- Hello, this is my project repo for my master degree of McGill University. I proposed a multi-object food detection architecture by deep convolutional neural networks (DCNN) with transferring features. In the world of computer vision, it could be pretty painful if you train everything from scratch. In my research, I pre-trained a food/non-food image classifier and then copy part of its weights to the food detectio neural networks.
- The overall architecture can be visualized like this: 

<img src="https://github.com/jianing-sun/Food-Detection-by-YOLOv2-with-Transfer-Learning/blob/master/asset/overall_method.png" />

Note the structure of *Feature Extraction Network* and *Food Detection Network* can be replaced by any CNN-based architecture as long as they have the same layer arrgement.

- Three state-of-the-art CNN architectures ([MobileNet](https://arxiv.org/pdf/1704.04861.pdf), [MobileNetV2](https://arxiv.org/pdf/1801.04381.pdf), [Resnet-18](https://arxiv.org/pdf/1512.03385.pdf)) have been implemented as backbones and evaluated. Below is the result contrasting the training process (loss history), mAP under different IoU (Intersection over Union) on datasets UECFood100 and UECFood256:

<img src="https://github.com/jianing-sun/Food-Detection-by-YOLOv2-with-Transfer-Learning/blob/master/asset/ablation_results.png"  />

- The effect of transfer learning has been quantified by designing the following experiments on MobileNet:

<img src="https://github.com/jianing-sun/Food-Detection-by-YOLOv2-with-Transfer-Learning/blob/master/asset/tl.png"  />

​	and the result as below: 

<img src="https://github.com/jianing-sun/Food-Detection-by-YOLOv2-with-Transfer-Learning/blob/master/asset/tfFig.png"  />

- Some results on food detection with MobileNetV2:

<img src="https://github.com/jianing-sun/Food-Detection-by-YOLOv2-with-Transfer-Learning/blob/master/asset/results.png"  />