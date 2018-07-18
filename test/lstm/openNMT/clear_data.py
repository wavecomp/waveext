#!/usr/bin/env python3

# clear_data.py 
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
#      Script for removing all files which are generated in training process,
#      and downloaded files for training and testing.
# 
# Author          : Nemanja Drobnjak
# Created On      : 05/08/2018
# 
#
import os
import glob
from tensorflow.python.platform import gfile

dir_name = "./model" 
if (gfile.Exists(dir_name)):
	for file in os.listdir(dir_name):
		file_path = os.path.join(dir_name, file)
		if os.path.isfile(file_path):
			os.remove(file_path)
	os.rmdir(dir_name)

dir_name = "./unsupervised-nmt-enfr-dev" 
if (gfile.Exists(dir_name)):
	for file in os.listdir(dir_name):
		file_path = os.path.join(dir_name, file)
		if os.path.isfile(file_path):
			os.remove(file_path)
	os.rmdir(dir_name)

dir_name = "./__pycache__" 
if (gfile.Exists(dir_name)):
	for file in os.listdir(dir_name):
		file_path = os.path.join(dir_name, file)
		if os.path.isfile(file_path):
			os.remove(file_path)
	os.rmdir(dir_name)	

for filename in glob.glob("scores-*"):
    os.remove(filename)

for filename in glob.glob("multi-bleu.perl*"):
    os.remove(filename)