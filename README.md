# Solar-deeplearning
This repository is an improvement method of [Deep-Solar-Eye](https://deep-solar-eye.github.io/)
## Background
  As the photovoltaic (PV) power has a very low carbon footprint, the use of solar panels is becoming widespread. However, the soiling of solar panels caused by severe weather will reduce up to 50% power generations. This challenge is considered by an existing method for quantifying the solar power loss. Whereas this method utilized a classification method, which is not sufficient for quantification resolution. To solve this, this project makes contribution on modifying the classification problem to a quantile regression problem based on the convolution neural network (CNN), which will increase the resolution of the quantification result.
## Environment
### Dependencies
* [Numpy](https://numpy.org/)
* [Pytorch](https://pytorch.org/)
### IDE
This project compiled on Visual Studio 2019
## Usage

Cuda is used, please check if cuda can be used on running GPU: [Cuda support](https://developer.nvidia.com/cuda-gpus)
## Dataset
A first-of-its-kind dataset, Solar Panel Soiling Image Dataset, comprising of 45,754 images of solar panels with power loss labels. From [Deep-Solar-Eye](https://deep-solar-eye.github.io/)

