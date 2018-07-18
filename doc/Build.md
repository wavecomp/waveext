# Building waveflow

The waveflow repository is partly in Python, partly in C++ (like Tensorflow). The Python parts don't need a build step. The C++ code however does have a build step. The required build process is:

```
>cd $REPO_TOP
>mkdir build
>cd build
>cmake -DCMAKE_BUILD_TYPE=Debug ..
>make all
```

waveflow does not require a specific compiler. It will eventually require the same compiler and version as the main Tensorflow distribution, but currently this is not required. If you do want a specific compiler, use the CC/CXX environment variables to specify it:

```
>export CC=clang
>export CXX=clang++
```

## Environment
Users should add the git repository directory to their PYTHONPATH. If the repository directory is `<repo_dir>`, then this can be done as:
```
>export PYTHONPATH=${PYTHONPATH}:<repo_dir>
```


## OS Dependencies
waveflow requires a lot of the common operating system build packages to be installed. Package names are OS-dependent, but the Wave common environment assumes Ubuntu 16.04 (and onward).

## Python Dependencies
waveflow is a Python 3 project; Python 2 is explicitly unsupported. Python package dependencies are listed in the `requirements.txt` file at the top of the repo. This file is always up to date, and is used by the Jenkins automated testing process each time a regression is run. Running `pip3 --user -r requirements.txt` will install all Python dependencies needed for the project.

## CMake
waveflow uses CMake for building the C++ source files. Either gcc or clang can be used as the compiler. waveflow will use a **single CMakefile** at the top level to build all project sources. Developers can put module-specific CMakefiles down in the repo branches, but those must be explicitly added in the top level CMakefile (via `add_subdirectory()`). The intent is to have a single build command for all files in the repository, which greatly simplifies code improvement and prevents out-of-date problems during development.
