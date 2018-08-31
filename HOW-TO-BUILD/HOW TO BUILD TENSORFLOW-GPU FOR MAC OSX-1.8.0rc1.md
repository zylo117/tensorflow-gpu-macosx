# HOW TO BUILD TENSORFLOW-GPU FOR MacOS 10.13.6 with CUDA

*** notice ***

1.INSTALL NVIDIA DRIVER

2.INSTALL NVIDIA CUDA TOOLKIT (9.2 OR LATER)

3.INSTALL NVIDIA CUDA CUDNN (7.2 OR LATER)

4.SET UP CUDA ENVIRONMENT (MAKE SURE 'nvcc -V' WORKS AND PRINTS CUDA VERSION)

5.INSTALL XCODE/COMMAND LINE TOOL 9.4 or later

6.INSTALL HOMEBREW

7.INSTALL COREUTILS USING 'brew install coreutils'

***8.INSTALL LLVM USING 'brew install llvm'

***9.INSTALL OPENMP USING 'brew install cliutils/apple/libomp'

***10.INSTALL BAZEL _0.16.1_ FROM GITHUB(https://github.com/bazelbuild/bazel/releases, newer/older version may cause failure)

***11.INSTALL XCODE/COMMAND LINE TOOL 8.2 (TF ONLY SUPPORTS XCODE 8 and 'sudo xcode-select --switch /Library/Developer/CommandLineTools')

12.GIT CLONE TENSORFLOW r1.10

13.CD TENSORFLOW DIR

14.MODIFY SOURCE USING(

    git apply THE/PATH/TO/tensorflow-1.10.1.patch

)

(prior copy of nccl.h now part of the patchfile)

15.CONFIG AND BUILD(

    ./configure
      #You have bazel 0.16.1 installed.
      #Please specify the location of python. [Default is /usr/local/opt/python@2/bin/python2.7]: /usr/local/bin/python3


      #Found possible Python library paths:
        /usr/local/Cellar/python/3.7.0/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages
      #Please input the desired Python library path to use.  Default is [/usr/local/Cellar/python/3.7.0/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages]

      #Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: n
      #Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: n
      #Do you wish to build TensorFlow with Amazon S3 File System support? [Y/n]: n
      #Do you wish to build TensorFlow with XLA JIT support? [y/N]: n
      #Do you wish to build TensorFlow with GDR support? [y/N]: n
      #Do you wish to build TensorFlow with VERBS support? [y/N]: n
      #Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: n
      #Do you wish to build TensorFlow with CUDA support? [y/N]: y
      #Please specify the CUDA SDK version you want to use, e.g. 7.0.: 9.2
      #Please specify the location where CUDA 9.2 toolkit is installed.: Accept the default option
      #Please specify the cuDNN version you want to use.: 7.2
      #Please specify the location where cuDNN 7 library is installed.: Accept the default option
      ##Please specify a list of comma-separated Cuda compute capabilities you want to build with.
      ##You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus. (GTX10X0: 6.1, GTX9X0: 5.2)
      #Please note that each additional compute capability significantly increases your build time and binary size.: 3.0, 6.1
      #Do you want to use clang as CUDA compiler? [y/N]: n
      #Please specify which gcc should be used by nvcc as the host compiler.: Accept the default option
      #Do you wish to build TensorFlow with MPI support? [y/N]: n
      #Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified:  Accept the default option
      #Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n

    export CUDA_HOME=/usr/local/cuda
    export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib
    export LD_LIBRARY_PATH=$DYLD_LIBRARY_PATH
    export PATH=$DYLD_LIBRARY_PATH:$PATH
    export BAZEL_USE_CPP_ONLY_TOOLCHAIN=1
    #bazel clean --expunge

    bazel build --config=cuda --config=opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --action_env PATH --action_env LD_LIBRARY_PATH --action_env DYLD_LIBRARY_PATH //tensorflow/tools/pip_package:build_pip_package


)

16.BUILD PYTHON BINDING USING 'bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg'

17.INSTALL PYTHON WHEEL USING 'pip3 install /tmp/tensorflow_pkg/*.whl'

18.REINSTALL XCODE/COMMAND LINE TOOL 9
