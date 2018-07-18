# Getting Started with Waveflow
## Summary
Waveflow is a supplemental Python package to Tensorflow (TF). It's designed to be useful with a minimum impact on user Python code. It can implement Dynamic Fixed Point (DFX) versions of significant Tensorflow operations, and also includes support for a lot of user-designed binary point (BP) management schemes. This document is intended to quickly describe these capabilities, as well as the preferred TF coding style for tests and examples in the Waveflow repository.

Waveflow is designed to be used for Research & Development prototyping, as well as accelerating models on the Wave DPU. DPU acceleration is still a work in progress and is not currently enabled. This will change in future releases.

## Using
Users will write their model in either pure Tensorflow or Keras. Both levels of abstraction are supported. However, if you want to write custom binary point management operations, only the pure TF API is supported. To run a model in the default DFX mode, the only thing required in the Python is
```
import waveflow
```
This import should be done after `import tensorflow`. By default, the appropriate operators are substituted to DFX versions.

Waveflow also has the notion of a user-selectable arithmetic type. Users can select this arithmetic on a model-wide basis by adding the following line
```
waveflow.waveflow_arithmetic = 'dfx'
```
This tells Waveflow to use DFX arithmetic for graph computation. Any supported DFX operator will be used when the user instantiates a corresponding op in the computation graph. There are 3 valid arithmetic settings in Waveflow: 
 1. `tf`: Use default Tensorflow operations
 1. `wf`: Use FP32 Waveflow operations. This mode is typically used only during development and testing. It provides no functional benefits over `tf`.
 1. `dfx`: Use DFX arithmetic.

## BP Management
### Automatic BP
Users can instantiate DFX operations manually. One example is
```
with tf.Session('') as sess:
    # Do your starting initialization (your code)
    t_init.run()
        
    z_op = waveflow.wavecomp_ops_module.wave_conv2d_dfx(activations, c2d_wts, strides=[1, stride, stride, 1], padding=padding)
```
This example instantiates a DFX version of the 2D convolution operator. It has an I/O signature identical to the built-in TF Conv2D operator. Note that this is actually optional; if the user does `import waveflow` at the beginning of the code, any implemented DFX operation will automatically be substituted in for built-in TF operations.

When these substitutions are made, Waveflow will compute an "ideal" binary point for all input and output tensors. Ideal mode means that a range-analysis is done inline with computation, and the binary point is chosen to represent the maximum value in the tensor, with as many bits as possible for the fraction. This method is computationally more expensive, but is guaranteed to provdide a near-ideal representation for fixed-point tensors. This is the default method because it is arithmetically very high quality, and represents an upper-bound on accuracy for fixed-point arithmetic. Users are empowered to write alternate methods of BP management, to improve computational efficiency.

### User-Directed BP
Users will probably want to experiment with different approaches to set the binary point for DFX arithmetic. To do this, there is a particular mechanism provided in Waveflow to generate a BP for a set of consuming ops, which you can specify. This capability looks as follows
```
with tf.Session('') as sess:
    # Do your starting initialization (your code)
    t_init.run()
            
    # The op output is the BP. These binary point values will be automatically bound to 
    # the name of the op itself.
    wts_bp = waveflow.wavecomp_ops_module.wave_bp_gen(c2d_wts)
    act_bp = waveflow.wavecomp_ops_module.wave_bp_gen(activations)

    # On the consumer side, we bind the BP of each input to the binary point generating
    # ops. We also use a control dependency to enforce proper ordering of execution.
    with tf.control_dependencies([act_bp, wts_bp]):
        z_op = waveflow.wavecomp_ops_module.wave_conv2d_dfx(activations, c2d_wts, strides=[1, 1, 1, 1], padding='SAME',
            bp_i0=act_bp.op.name, bp_i1=wts_bp.op.name)
```
This code shows an example of generating a custom BP for a 2D convolution operator. The op which generates the BP is called `wave_bp_gen`, and in this case is provided as a standard element of the Waveflow library. It generates 1 outputs. The output is a 1D, 2-element tensor containing the (word length, binary point) computed by the operation. It is provided as a primary output of the op so that users may log this value over time, or use it as an input for other computation in the graph. It is not strictly required to consume it however.

This example has one consuming operator, a 2D convolution. This operation is a DFX variant, and it specifies that there is a generated binary point for input tensors 1 & 2. The attribute `bp_i0` indicates that a binary point for input tensor 1 should be taken from BP generating op `act_bp`. This attribute must be associated with a BP producing operation, or else you will get an exception when you try and run the model. BP values can be specified for all input tensors if desired, or only select inputs. Note that the Tensorflow control dependency capability is used. This is necessary to properly order the BP generation ops and the consuming operators. This will be necessary for every pairing of generator and consuming op.

## Testing
Waveflow has a comprehensive set of tests which are run everytime there is a checkin. If you implement a new binary point operator, you are encouraged to submit a pull request to have it included in the master git branch. One requirement for this is to have at least one Python test which exercises this operation to ensure it passes some basic criteria. The Waveflow directory `test/` contains unit tests for the project, and this is the place to put your test. There is another document called `Developers_Guide.md` which describes accepted practices for contributing to Waveflow, which you should read if you like to contribute operators.

In addition, all the tests under the `test/` directory are excellent references for how to write Waveflow code. There are also examples included, which are identified with an `example_` prefix on the filename. They are fully-functional DNN models which can be used as a basis for experiments or development.


