import datetime
import fcntl
import os
import subprocess
import sys
import time

from configure import run_shell

pwd = sys.path[0]  # make sure work dir to here


# abandoned
def pre_config():
    cuda_version = subprocess.Popen("nvcc -V", shell=True, stdout=subprocess.PIPE)
    cuda_version = cuda_version.stdout.read().decode()
    if '9.1' in cuda_version:
        CUDA_VERSION_MATCHES = True
        print('CUDA version matches')

    if CUDA_VERSION_MATCHES:
        coreutils = subprocess.Popen("brew install coreutils", shell=True, stdout=subprocess.PIPE)
        coreutils_text = coreutils.stdout.read().decode()
        if 'already installed' in coreutils_text:
            print(coreutils_text)


def get_python_path():
    return subprocess.Popen('which python3', shell=True, stdout=subprocess.PIPE).stdout.read().decode().strip(
        '\n').strip('\r')


def get_python_package():
    for p in sys.path:
        if 'site-packages' in p:
            if p.split('/')[-1] == 'site-packages':
                return p


def config(python_exec, python_lib_path):
    conf = open(pwd + '/.tf_configure.bazelrc', 'r')
    conf_data = conf.readlines()
    conf_data[0] = conf_data[0].strip('\n').split('=')[0] + '=' + '"%s"' % python_exec + '\n'
    conf_data[1] = conf_data[1].strip('\n').split('=')[0] + '=' + '"%s"' % python_lib_path + '\n'
    conf = open(pwd + '/.tf_configure.bazelrc', 'w')
    conf.writelines(conf_data)
    conf.close()


def build_bazel():
    bazel = subprocess.Popen(
        'bazel build --config=cuda --config=opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --action_env PATH --action_env LD_LIBRARY_PATH --action_env DYLD_LIBRARY_PATH //tensorflow/tools/pip_package:build_pip_package',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, bufsize=128)

    fd = bazel.stderr.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

    while True:
        try:
            line = bazel.stderr.readline().decode().strip('\n')
            if len(line) > 0:
                print(line)
            if 'Build completed successfully' in line:
                break
        except:
            time.sleep(1)


def build_pip():
    pip = subprocess.Popen('bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg', shell=True,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, bufsize=128)
    fd1 = pip.stdout.fileno()
    fl1 = fcntl.fcntl(fd1, fcntl.F_GETFL)
    fcntl.fcntl(fd1, fcntl.F_SETFL, fl1 | os.O_NONBLOCK)

    fd2 = pip.stderr.fileno()
    fl2 = fcntl.fcntl(fd2, fcntl.F_GETFL)
    fcntl.fcntl(fd2, fcntl.F_SETFL, fl2 | os.O_NONBLOCK)
    while True:
        try:
            line1 = pip.stdout.readline().decode().strip('\n')
            line2 = pip.stderr.readline().decode().strip('\n')
            if len(line1) > 0:
                print(line1)
            if len(line2) > 0:
                print(line2)
            if 'Output wheel file' in line1 or 'Output wheel file' in line2:
                break
        except:
            time.sleep(1)


def install_pip(pip_path):
    testdir = '/tmp/tensorflow_pkg/'
    list = os.listdir(testdir)
    list.sort(key=lambda fn: os.path.getmtime(testdir + '/' + fn))
    filetime = datetime.datetime.fromtimestamp(os.path.getmtime(testdir + list[-1]))
    filepath = os.path.join(testdir, list[-1])
    install = subprocess.Popen('%s install %s' % (pip_path,filepath),
                           shell=True,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, bufsize=128)

    fd1 = install.stdout.fileno()
    fl1 = fcntl.fcntl(fd1, fcntl.F_GETFL)
    fcntl.fcntl(fd1, fcntl.F_SETFL, fl1 | os.O_NONBLOCK)

    fd2 = install.stderr.fileno()
    fl2 = fcntl.fcntl(fd2, fcntl.F_GETFL)
    fcntl.fcntl(fd2, fcntl.F_SETFL, fl2 | os.O_NONBLOCK)
    while True:
        try:
            line1 = install.stdout.readline().decode().strip('\n')
            line2 = install.stderr.readline().decode().strip('\n')
            if len(line1) > 0:
                print(line1)
            if len(line2) > 0:
                print(line2)
            if 'Successfully installed' in line1 or 'Successfully installed' in line2:
                break
        except:
            time.sleep(1)


if __name__ == '__main__':
    print(get_python_path())
    print(get_python_package())
