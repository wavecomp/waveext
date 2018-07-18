#!/usr/bin/env python3

# test_config.py
#
# Copyright (c) 2010-2018 Wave Computing, Inc. and its applicable licensors.   
# All rights reserved; provided, that any files identified as open source shall
# be governed by the specific open source license(s) applicable to such files. 
#
# For any files associated with distributions under the Apache 2.0 license, 
# full attribution to The Apache Software Foundation is given via the license 
# below.
#
# PURPOSE
#      Top level run script for regression testing.
#
# Author          : Nemanja Popov
# Created On      : 03/13/2018
#
 # Licensed under the Apache License, Version 2.0 (the "License");
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at
 # 
 #     http://www.apache.org/licenses/LICENSE-2.0
 # 
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License.

import tensorflow as tf

import waveflow

def test_config():
    # set first and third config value
    config1 = waveflow.wavecomp_ops_module.wave_config(rounding_mode = "stochastic", config3 = "cfg3")
    tf.Session().run(config1)

    # add second config value without changing previous setup
    config2 = waveflow.wavecomp_ops_module.wave_config(config2 = "new_cfg2")
    tf.Session().run(config2)

    return True

if __name__ == "__main__":
    test_config()