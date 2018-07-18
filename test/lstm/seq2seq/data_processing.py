#!/usr/bin/env python3

# data_processing.py 
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
#      Data pre- and post- processing functions for seq2seq example.
# 
# Author          : Ken Shiring
# Created On      : 04/12/2018
# 
#

import os, requests, zipfile
from urllib.parse import urlparse
import numpy as np


def download_dataset(url, dst_path):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    a = urlparse(url)
    file_name = os.path.basename(a.path)
    file_path = os.path.join(dst_path, file_name)

    r = requests.get(url, allow_redirects=True)
    open(file_path, 'wb').write(r.content)

    zfile = zipfile.ZipFile(file_path, 'r')
    extract_folder, _ = os.path.splitext(file_path)
    zfile.extractall(extract_folder)


class DataCreator(object):
    ''' This object abstracts away reading in the dataset, both the input and the
        target data.
    '''
    def __init__(self, data_path, num_samples):
        self._data_path = data_path
        self._num_samples = num_samples
        self._input_texts = []
        self._target_texts = []
        self._input_characters = set()
        self._target_characters = set()
        self._data_stats = {}

    def read_data(self):
        with open(self._data_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        for line in lines[: min(self._num_samples, len(lines) - 1)]:
            input_text, target_text = line.split('\t')
            # We use "tab" as the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
            target_text = '\t' + target_text + '\n'
            self._input_texts.append(input_text)
            self._target_texts.append(target_text)
            for char in input_text:
                if char not in self._input_characters:
                    self._input_characters.add(char)
            for char in target_text:
                if char not in self._target_characters:
                    self._target_characters.add(char)

        self._input_characters = sorted(list(self._input_characters))
        self._target_characters = sorted(list(self._target_characters))
        self._data_stats['num_encoder_tokens'] = len(self._input_characters)
        self._data_stats['num_decoder_tokens'] = len(self._target_characters)
        self._data_stats['max_encoder_seq_length'] = max([len(txt) for txt in self._input_texts])
        self._data_stats['max_decoder_seq_length'] = max([len(txt) for txt in self._target_texts])
        self._input_token_index = dict(
            [(char, i) for i, char in enumerate(self._input_characters)])
        self._target_token_index = dict(
            [(char, i) for i, char in enumerate(self._target_characters)])


    def print_summary(self):
        print('Number of samples:', len(self._input_texts))
        print('Number of unique input tokens:', self._data_stats['num_encoder_tokens'])
        print('Number of unique output tokens:', self._data_stats['num_decoder_tokens'])
        print('Max sequence length for inputs:', self._data_stats['max_encoder_seq_length'])
        print('Max sequence length for outputs:', self._data_stats['max_decoder_seq_length'])

    def get_encode_tokens(self):
        return self._data_stats['num_encoder_tokens']

    def get_decode_tokens(self):
        return self._data_stats['num_decoder_tokens']

    def get_max_decode_seq(self):
        return self._data_stats['max_decoder_seq_length']

    def get_input_char(self, index):
        return self._input_characters[index]

    def get_target_char(self, index):
        return self._target_characters[index]

    def get_target_index(self, c):
        return self._target_token_index[c]

    def get_input_text(self):
        return self._input_texts

    def onehot_encode(self):
        encoder_input_data = np.zeros(
            (len(self._input_texts), 
            self._data_stats['max_encoder_seq_length'], 
            self._data_stats['num_encoder_tokens']),
            dtype='float32')
        decoder_input_data = np.zeros(
            (len(self._input_texts), 
            self._data_stats['max_decoder_seq_length'], 
            self._data_stats['num_decoder_tokens']), 
            dtype='float32')
        decoder_target_data = np.zeros(
            (len(self._input_texts), 
            self._data_stats['max_decoder_seq_length'], 
            self._data_stats['num_decoder_tokens']), 
            dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(self._input_texts, self._target_texts)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, self._input_token_index[char]] = 1.
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t, self._target_token_index[char]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, self._target_token_index[char]] = 1.

        return (encoder_input_data, decoder_input_data, decoder_target_data)



class NetworkDecoder(object):
    ''' Helper class to use the trained models to translate text.
    '''
    def __init__(self, encoder_model, decoder_model, dataset):
        self._encoder_model = encoder_model
        self._decoder_model = decoder_model
        self._max_decoder_seq_length = dataset.get_max_decode_seq()
        self._num_decoder_tokens = dataset.get_decode_tokens()
        # Reverse-lookup token index to decode sequences back to
        # something readable.
        self._dataset = dataset
        # self._reverse_input_char_index = dict(
        #     (i, char) for char, i in input_token_index.items())
        # self._reverse_target_char_index = dict(
        #     (i, char) for char, i in target_token_index.items())

    def decode_sequence(self, input_seq):
        # Encode the input as state vectors.
        states_value = self._encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self._num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self._dataset.get_target_index('\t')] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self._decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self._dataset.get_target_char(sampled_token_index)
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
               len(decoded_sentence) > self._max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self._num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence

    def predict_samples(self, outfile, encoder_input_data):
        print("Writing sample prediction to %s" % (outfile))
        with open(outfile, 'w', encoding='utf-8') as f:
            ''' Predict samples and emit results to a file.
            '''
            for seq_index in range(100):
                # Take one sequence (part of the training set)
                # for trying out decoding.
                input_seq = encoder_input_data[seq_index: seq_index + 1]
                decoded_sentence = self.decode_sequence(input_seq)
                input_texts = self._dataset.get_input_text()
                f.write('-\n')
                f.write('Input sentence: %s\n' % (input_texts[seq_index]))
                f.write('Decoded sentence: %s\n' % (decoded_sentence))
            f.close()


