import custom_build_tools.build as build

if __name__ == '__main__':
    python_exec=build.get_python_path()
    python_lib_path=build.get_python_package()
    build.config(python_exec, python_lib_path)
    build.build_bazel()