import time, os, sys, shutil
import custom_build_tools.build as build_tools

pwd = sys.path[0]

# build TF
print('[INFO] Building Tensorflow...')
print("[INFO] It's going to last for about 90 minutes on Intel i7-6700")
# time.sleep(1)
python_exec = build_tools.get_python_path()
python_bin = '/'.join(build_tools.get_python_path().split('/')[:-1])
python_lib_path = build_tools.get_python_package()
build_tools.config(python_exec, python_lib_path)
build_tools.build_bazel()
print('[INFO] Build Complete')

# copy true setup.py into install dir
shutil.copyfile(pwd + '/.setup.py', pwd + '/tensorflow/tools/pip_package/setup.py')
print('[INFO] Building wheel')
build_tools.build_pip()
print('[INFO] Wheel Build Complete\n')
print('[INFO] Installing wheel')
build_tools.install_pip(python_bin + '/pip')
print('[INFO] Instal Complete')