#!/usr/bin/env python3

# example_lstm_seq2seq.py 
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
#      seq2seq example for English-French translation.
# 
# Author          : Nemanja Popov
# Created On      : 03/27/2018
# 
#
# Sequence to sequence example in Keras (character-level).
# This script demonstrates how to implement a basic character-level
# sequence-to-sequence model. We apply it to translating
# short English sentences into short French sentences,
# character-by-character. Note that it is fairly unusual to
# do character-level machine translation, as word-level
# models are more common in this domain.
# # Summary of the algorithm
# - We start with input sequences from a domain (e.g. English sentences)
#     and correspding target sequences from another domain
#     (e.g. French sentences).
# - An encoder LSTM turns input sequences to 2 state vectors
#     (we keep the last LSTM state and discard the outputs).
# - A decoder LSTM is trained to turn the target sequences into
#     the same sequence but offset by one timestep in the future,
#     a training process called "teacher forcing" in this context.
#     Is uses as initial state the state vectors from the encoder.
#     Effectively, the decoder learns to generate `targets[t+1...]`
#     given `targets[...t]`, conditioned on the input sequence.
# - In inference mode, when we want to decode unknown input sequences, we:
#     - Encode the input sequence into state vectors
#     - Start with a target sequence of size 1
#         (just the start-of-sequence character)
#     - Feed the state vectors and 1-char target sequence
#         to the decoder to produce predictions for the next character
#     - Sample the next character using these predictions
#         (we simply use argmax).
#     - Append the sampled character to the target sequence
#     - Repeat until we generate the end-of-sequence character or we
#         hit the character limit.
#
# # Data download
# English to French sentence pairs.
# http://www.manythings.org/anki/fra-eng.zip
# Lots of neat sentence pairs datasets can be found at:
# http://www.manythings.org/anki/
#
# # References
# - Sequence to Sequence Learning with Neural Networks
#     https://arxiv.org/abs/1409.3215
# - Learning Phrase Representations using
#     RNN Encoder-Decoder for Statistical Machine Translation
#     https://arxiv.org/abs/1406.1078
# 


from __future__ import print_function

import numpy as np
import os, requests, zipfile
import argparse
from urllib.parse import urlparse

from keras.models import Model
from keras.layers import Input, LSTM, Dense, BatchNormalization
from keras import backend as K
from keras.callbacks import TensorBoard
from tensorflow.python.platform import gfile


import waveflow
import data_processing


parser = argparse.ArgumentParser(description='Run seq2seq example')
parser.add_argument("-tb", "--tensorboard", action='store_true', help='Generate Tensorboard data')
parser.add_argument("-tr", "--trace", action='store_true', help='Generate execution trace')
parser.add_argument("-a",  "--arith", type=str, default='tf', help='Arithmetic type')

args = parser.parse_args()

waveflow.waveflow_arithmetic = args.arith

kwrap = waveflow.K_TBLogger(log_dir='./tb_s2s_log', enable_tb=args.tensorboard, enable_trace=args.trace, 
    unified_trace=True, arith_type=waveflow.waveflow_arithmetic)

batch_size = 64  # Batch size for training.
epochs = 30  # Number of epochs to train for.
# epochs = 1  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = './data/fra-eng/fra.txt'
data_url = 'http://www.manythings.org/anki/fra-eng.zip'
dst_folder = './data'

if not (gfile.Exists(data_path)):
   data_processing.download_dataset(data_url, dst_folder)


# Read in all the data, parse it out, and generate 1-hot numeric data for RNN training.
dataset = data_processing.DataCreator(data_path=data_path, num_samples=num_samples)
dataset.read_data()
dataset.print_summary()
encoder_input_data, decoder_input_data, decoder_target_data = dataset.onehot_encode()

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, dataset.get_encode_tokens()))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, dataset.get_decode_tokens()))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
dec_out2 = BatchNormalization()(decoder_outputs)
decoder_dense = Dense(dataset.get_decode_tokens(), activation='softmax')
# decoder_outputs = decoder_dense(decoder_outputs)
decoder_outputs = decoder_dense(dec_out2)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

print('nodes before trainng')
print('op list: ', waveflow.op_list(K.get_session().graph))

# Run training
kwrap.compile(model, optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

kwrap.fit(model, [encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)

kwrap.close()

# Save model
model.save('seq2seq_model.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


# Inference testing
sample_decoder = data_processing.NetworkDecoder(encoder_model, decoder_model, dataset)
sample_decoder.predict_samples(encoder_input_data=encoder_input_data, outfile='sample.txt')


