# Testing waveflow

waveflow is designed for a simple testing approach. Tests should be run by developers when features are added/bugs fixed before all checkins. To run the regression locally, run:
```
>cd test/
>pytest
```
pytest should show a green bar at the bottom if all tests pass. To learn more about pytest, see the [excellent online reference](https://docs.pytest.org/en/latest/index.html).

To run an individual test, you can do it 2 ways:
`>pytest -x test_file.py`
or simply
`>./test_file.py`
The 2nd variant is for running a test without the pytest environment; that can be very useful for interactive debugging and seeing lots of debug output. Please see the test `test_config.py` to see a very simple test from which you can build your own.

## Adding Tests
New tests should be Python at the top level. This allows for easy integration with pytest. Tests should be added in the test/ directory, and be named pytest-style, "test_<something>.py". Tests need to be self-checking as much as possible. Extensive use of assertions is encouraged. Broad code coverage is highly desired, and developers are expected to add as many tests as possible to ensure correctness and quality for any code in the waveflow repository. For the time being, developers have no QA support, and must be self-serving when it comes to testing.
