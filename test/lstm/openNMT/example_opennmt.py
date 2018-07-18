#!/usr/bin/env python3

# train.py.py 
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
#      Script for training of the model
# 
# Author          : Nemanja Drobnjak
# Created On      : 05/08/2018
# 
#
# We created complete training algorithm with training and inference Python scripts.

from __future__ import print_function

import numpy as np
import os, requests, zipfile
import argparse
from urllib.parse import urlparse
import time

from tensorflow.python.platform import gfile

import waveflow

import wget
import tarfile
from training import train
from inference import inf
import tensorflow as tf

data_dir= './unsupervised-nmt-enfr-dev/'
	
src_vocab= data_dir + 'en-vocab.txt'
tgt_vocab= data_dir + 'fr-vocab.txt'

src= data_dir + 'train.en.10k'
tgt= data_dir + 'train.fr.10k'
src_trans= data_dir + 'train.en.10k.m1'
tgt_trans= data_dir + 'train.fr.10k.m1'

src_test= data_dir + 'newstest2014.en.tok'
tgt_test= data_dir + 'newstest2014.fr.tok'
src_test_trans= data_dir + 'newstest2014.en.tok.m1'
tgt_test_trans= data_dir + 'newstest2014.fr.tok.m1'

data_path = data_dir + 'en-vocab.txt'
data_url = 'https://s3.amazonaws.com/opennmt-trainingdata/unsupervised-nmt-enfr-dev.tar.bz2'
perl_file_path = 'https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl'


def download_dataset():

  data_set = wget.download(data_url)
  print("\nData set is downloaded ")
  tar = tarfile.open("unsupervised-nmt-enfr-dev.tar.bz2", "r:bz2")
  tar.extractall()
  tar.close()
  print("Data set is extracted ")
  os.remove('unsupervised-nmt-enfr-dev.tar.bz2')
  print("tar.bz2 file is deleted ")
  perl_file = wget.download(perl_file_path)
  print("\nPerl file is downloaded ")

def score_test():
  command_perl = 'perl multi-bleu.perl ' + tgt_test + ' < ' + src_test_trans + ' >> ' + score_file
  os.system(command_perl)
  command_perl = 'perl multi-bleu.perl ' + src_test + ' < ' + tgt_test_trans + ' >> ' + score_file
  os.system(command_perl)

if not (gfile.Exists(data_path)):
   download_dataset()

current_milli_time = lambda: int(round(time.time() * 1000))
timestamp = current_milli_time()
score_file= 'scores-' + str(timestamp) + '.txt'

score_test()

if __name__ == "__main__":
    waveflow.waveflow_arithmetic = "dfx"
    for i in range(2, 5):
      # Train for one epoch.
      train(src, tgt, src_trans, tgt_trans, src_vocab, tgt_vocab)
      tf.reset_default_graph()

      src_test_trans = src_test + '.m' + str(i)
      tf.reset_default_graph()
      tgt_test_trans = tgt_test + '.m' + str(i)
      tf.reset_default_graph()

      inf(src, tgt, src_vocab, tgt_vocab, '1', src_test_trans)
      tf.reset_default_graph()
      inf(src, tgt, src_vocab, tgt_vocab, '2', src_test_trans)
      tf.reset_default_graph()

      score_test()

      # Translate training data.
      src_trans = src + '.m' + str(i)
      tgt_trans = tgt + '.m' + str(i)

      inf(src, tgt, src_vocab, tgt_vocab, '1', src_test_trans)
      tf.reset_default_graph()
      inf(src, tgt, src_vocab, tgt_vocab, '2', src_test_trans)
      tf.reset_default_graph()
      
