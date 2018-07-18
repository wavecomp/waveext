# Rebuild and install of the modified Tensorflow

** Building pip package **

> cd tensorflow/
> bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
> bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

Running this will build a .whl file within the /tmp/tensorflow_pkg directory

** Instaling pip package **
This step can be done native or in your virtual environment.
For native and in Virtualenv environment:
> pip install --upgrade tfBinaryURL

In Anaconda:
> pip install --ignore-installed --upgrade tfBinaryURL

, where "tfBinaryURL" identifies the URL of the TensorFlow Python package, made with previous step in /tmp/tensorflow_pkg directory, for example:

> pip install --upgrade /tmp/tensorflow_pkg/tensorflow-1.5.0-cp35-cp35m-linux_x86_64.whl



For more information, check: 
https://www.tensorflow.org/install/install_sources#build_the_pip_package
https://www.tensorflow.org/install/install_linux#determine_how_to_install_tensorflow
