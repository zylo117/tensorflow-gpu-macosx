# tensorflow-gpu-macosx

Unoffcial NVIDIA CUDA GPU support version of Google Tensorflow for MAC OSX 10.13

Description:

    Since Google Tensorflow claimed that tensorflow-gpu no longer supports 
    MAC OSX since 1.2.0 due to the OpenMP issue of clang of Apple,
    I built this unoffcial tensorflow-gpu for MAC OSX so that
    Hackintosh users or Mac users with eGPU can run tensorflow-gpu with CUDA.

NOTICE:

    This version will not supports multi-cpu/machine training.
    
    It's not like you are gonna need it anyway.

INSTALL:

    pip3 install tensorflow-gpu-macosx

Requirement:

    1. Must use python3 instead of python2

    2. Must be installed on MAC OSX 10.6+

    3. Must install Nvidia GPU drivers

    4. Must install Nvidia CUDA toolkit 9.1

    5. Must install Nvidia CUDA cudnn 7.0

    6. Must set up cuda environment (make sure 'nvcc -V' shows the cuda version '9.1')

Please check out more details on: https://github.com/zylo117/tensorflow-gpu-macosx