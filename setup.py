from distutils.core import setup

from setuptools import find_packages

__version__ = '1.8.0'

REQUIRED_PACKAGES = [
    'absl-py >= 0.1.6',
    'astor >= 0.6.0',
    'gast >= 0.2.0',
    'grpcio >= 1.8.6',
    'numpy >= 1.13.3',
    'scipy >=0.15.1',
    'protobuf >= 3.4.0',
    'six >= 1.10.0',
    'tensorflow-tensorboard >= 1.5.1',
    'termcolor >= 1.1.0',
    'wheel >= 0.26',
]

setup(name='tensorflow-gpu-macosx',
      version=__version__,
      description='Unoffcial NVIDIA CUDA GPU support version of Google Tensorflow for MAC OSX 10.13',
      author='Carl Cheung',
      author_email='zylo117@hotmail.com',
      url='https://github.com/zylo117/tensorflow-gpu-macosx',
      packages=find_packages(),
      # Contained modules and scripts.
      install_requires=REQUIRED_PACKAGES,
      # Add in any packaged data.
      include_package_data=True,
      package_data={'': ['*.so']},
      exclude_package_data={'': ['BUILD', '*.h', '*.cc']},
      zip_safe=False,
      # PyPI package information.
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Software Development :: Libraries :: Python Modules',
          'Topic :: Software Development :: Libraries',
          'Operating System :: MacOS :: MacOS X',
      ],
      license='Apache 2.0',
      keywords='tensorflow tensor machine learning gpu mac osx cuda nvidia',
      )
