# Winograd_Convolution
Winograd_Convolution is a winograd based kernel for convolutions in deep learning frameworks, which is an implementation of winograd convolutions in [1]. Three WT methods,  WT_6X6_F_4X4_3X3, WT_8X8_F_4X4_5X5, and WT_8X8_F_6X6_3X3, are supported, where convolution kernel 3x3 is the best choice. Parts of this work are from SkimCaffe [2], but this winograd kernel is more portable.

Dependencies
-----------------------------------

A fast blas is better, such as mkl-gemm and openblas [3].

Building
-----------------------------------

Only header files written in C++, supports windows and linux. This version is built on VS 2015.

Testing
-----------------------------------

See winograd_test.cpp.

Packaging
-----------------------------------

"include/winograd_layer.h", can be natively integrated into some famous deep learning frameworks as a winograd_layer, like caffe (https://github.com/BVLC/caffe) and tiny-dnn (https://github.com/tiny-dnn/tiny-dnn).

References & Dependencies
-----------------------------------
[1] Andrew Lavin, Scott Gray. Fast Algorithms for Convolutional Neural Networks. https://arxiv.org/abs/1509.09308

[2] SkimCaffe, https://github.com/IntelLabs/SkimCaffe

[3] OpenBLAS, https://github.com/xianyi/OpenBLAS.

License
-----------------------------------
The BSD 3-Clause License
