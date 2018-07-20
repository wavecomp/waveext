# waveflow
waveflow is a software project from Wave Computing to provide a Tensorflow-compatible programming interface to run machine learning models on the Wave DPU.

waveflow is designed so that users can take an existing Tensorflow model and run it with little or no modification on Wave Computing DPU products. waveflow will take care of all complexity of supporting DPUs, such that users won't have to be concerned about any technical differences. This includes transparent support for Wave's Dynamic Fixed Point (DFX) arithmetic when machine learning models are run.

waveflow has a direct dependency on Google's Tensorflow framework. Accordingly, waveflow is designed to match with a specific, very recent release-class version of Tensorflow. While there may be a development branch of waveflow to match deverloper milestones of Tensorflow, the waveflow will always be paired with a stable release of Tensorflow.
