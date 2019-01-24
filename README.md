# tensorflow-gpu-macosx

Unoffcial NVIDIA CUDA GPU support version of Google Tensorflow 1.8 for MAC OSX 10.13

**Description:**

    Since Google Tensorflow claimed that tensorflow-gpu no longer supports 
    MAC OSX since 1.2.0 due to the OpenMP issue of clang of Apple,
    I built this unoffcial tensorflow-gpu for MAC OSX so that
    Hackintosh users or Mac users with eGPU can run tensorflow-gpu with CUDA.

**Requirement:**

    1. Must use python3 instead of python2

    2. Must be installed on MAC OSX 10.6+

    3. Must install Nvidia GPU drivers

    4. Must install Nvidia CUDA toolkit 9.1 (if not, you need to re-compile by yourself)

    5. Must install Nvidia CUDA cudnn 7.0 (if not, you need to re-compile by yourself)

    6. Must set up cuda environment (make sure 'nvcc -V' shows the cuda version '9.1')

    7.CUDA compute capability is in [3.0,5.0] (if not, you need to re-compile to get a better performance)

# HOW TO BUILD TENSORFLOW-GPU FOR MAC OSX?

## NOTICE 

1.INSTALL NVIDIA DRIVER

2.INSTALL NVIDIA CUDA TOOLKIT (9.1 OR LATER)

3.INSTALL NVIDIA CUDA CUDNN (7.0 OR LATER)

4.SET UP CUDA ENVIRONMENT (MAKE SURE

    nvcc -V

WORKS AND PRINTS CUDA VERSION)

5.INSTALL XCODE/COMMAND LINE TOOL 9.3+

6.INSTALL HOMEBREW

7.INSTALL COREUTILS USING 

    brew install coreutils

**8.INSTALL LLVM USING**

    brew install llvm
    
**9.INSTALL OPENMP USING** 

    brew install cliutils/apple/libomp

**10.INSTALL BAZEL 0.16.1 FROM GITHUB**
*(https://github.com/bazelbuild/bazel/releases, newer/older version may cause failure)*

12.GIT CLONE TENSORFLOW

    git clone https://github.com/zylo117/tensorflow-gpu-macosx

13.CD TENSORFLOW SOURCE DIR

14.CONFIG(skip this step if your CUDA version is same as mine)

    ./configure
      #Please specify the location of python.: Accept the default option
        #Please input the desired Python library path to use.:  Accept the default option
        #Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: n
        #Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: n
        #Do you wish to build TensorFlow with Amazon S3 File System support? [Y/n]: n
        #Do you wish to build TensorFlow with XLA JIT support? [y/N]: n
        #Do you wish to build TensorFlow with GDR support? [y/N]: n
        #Do you wish to build TensorFlow with VERBS support? [y/N]: n
        #Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: n
        #Do you wish to build TensorFlow with CUDA support? [y/N]: y
        #Please specify the CUDA SDK version you want to use, e.g. 7.0.: 9.1
        #Please specify the location where CUDA 9.1 toolkit is installed.: Accept the default option
        #Please specify the cuDNN version you want to use.: 7
        #Please specify the location where cuDNN 7 library is installed.: Accept the default option
        ##Please specify a list of comma-separated Cuda compute capabilities you want to build with.
        ##You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus. (GTX10X0: 6.1, GTX9X0: 5.2)
        #Please note that each additional compute capability significantly increases your build time and binary size.: 6.1,5.2,5.0,3.0
        #Do you want to use clang as CUDA compiler? [y/N]: n
        #Please specify which gcc should be used by nvcc as the host compiler.: Accept the default option
        #Do you wish to build TensorFlow with MPI support? [y/N]: n
        #Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified:  Accept the default option
        #Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n

    export CUDA_HOME=/usr/local/cuda
    export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib
    export LD_LIBRARY_PATH=$DYLD_LIBRARY_PATH
    export PATH=$DYLD_LIBRARY_PATH:$PATH

    # bazel clean --expunge # Use this if you failed to compile before.

**15.BUILD**

    bazel build --config=cuda --config=opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --action_env PATH --action_env LD_LIBRARY_PATH --action_env DYLD_LIBRARY_PATH //tensorflow/tools/pip_package:build_pip_package

**16.NCCL PATCH**
  
  You can compile NCCL_OPS by yourself (not necessary):

    gcc -c -fPIC ./nccl_patched/nccl_ops.cc -o ./nccl_patched/_nccl_ops.o
    
    gcc ./nccl_patched/_nccl_ops.o -shared -o ./nccl_patched/_nccl_ops.so
  
  Then replace the original nccl lib:

    mv ./bazel-bin/tensorflow/contrib/nccl/python/ops/_nccl_ops.so ./bazel-bin/tensorflow/contrib/nccl/python/ops/_nccl_ops.so.bk

    cp ./nccl_patched/_nccl_ops.so ./bazel-bin/tensorflow/contrib/nccl/python/ops/



16.BUILD PYTHON BINDING USING

    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package/ ./tmp/tensorflow_pkg

17.INSTALL PYTHON WHEEL USING 'pip3 install /tmp/tensorflow_pkg/*.whl'

18.REINSTALL XCODE/COMMAND LINE TOOL 9


**NOTICE**

    This version will not supports multi-cpu/machine training.
    
    It's not like you are gonna need it anyway.

**INSTALL:**

    **Either compile from src through pypi (not recommended)**
        pip3 install tensorflow-gpu-macosx
    **or install from relese (the easiest way)**
        pip3 install *.whl
