
# Waveflow Developer's Guide
Waveflow is a supplemental package to Google's Tensorflow. It is designed to facilitate Wave R&D activity, as well as serve as a heterogeneous device platform for the Wave DPU. Please ensure you have read the other provided documents in this repository, so you can understand the specific goals and requirements of the Waveflow project.

## Repository Organization
This repo has an organization that is based on Python project best practice. Python project organization is described really well [here](http://docs.python-guide.org/en/latest/writing/structure/). All source code that is internal to Waveflow should be under the `waveflow/` directory. Under this directory, there are subdirectories by major component. Operator kernels, for example, live under `waveflow/kernel_lib/`.

## Building
The included file Build.md contains instructions on how to build Waveflow.

## Dependencies
The included file Build.md contains information on the dependencies inside Waveflow. It's very important that you ensure these dependencies are met in your own environment first before developing any new code in Waveflow.

## Code Hygeine
There are two languages used inside the Waveflow repository, C\++ and Python3. Each language has a coding standard that needs to be followed by developers.

### C++
All C++ code should be C\++14 compliant. The main reason for this specific standard is that Tensorflow itself follows the '14 standard, and Waveflow code interacts heavily with internal TF C\++ code. Coding style should follow [Google's published style guide](https://google.github.io/styleguide/cppguide.html). This style guide contains lexical style as well as feature dos/donts. Unless specifically mentioned, all do/dont rules in the Google Guide are followed here. Notable items include:
1. Don't use Exceptions
1. Do make use of the _auto_ keyword and other helpful C++14 features.
1. Do take steps to keep any persistent state as local as possible to each object.
1. Do use snake-case naming, and don't use camel-case.

### Python
All Python code should assume Python3 as the standard, and it will be the only tested version. Python2 has officially sunset, and nearly all public Python projects are migrating to Python3. Python2 compatibility is not desired, so there is no need to use helpers like the [six module](https://pypi.org/project/six/). All Python code should follow the [PEP-8 style](https://www.python.org/dev/peps/pep-0008/).

For the Waveflow project, the preference is to use Python anywhere it's possible to use, rather than C\++. Development speed is much faster, and bug avoidance will be improved. There are some improvements which can only be done in C\++ (e.g. operator kernels), but everything else should favor a Python solution before implementing a C\++ solution.

## Testing
The provided Testing.md document describes the approach to testing in Waveflow. Unit testing is a requirement, and developers should implement their own unit testing. Additional testing will be provided by QA personnel, but all new code should be tested by the development staff first.
