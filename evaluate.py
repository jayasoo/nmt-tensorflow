#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 14:52:59 2019

@author: jayasoo
"""
import os
import pickle
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from models import Encoder, Decoder
from beam_search import beam_search
import constants
from preprocessing import preprocess_text

tf.enable_eager_execution()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def evaluate(text):
    with open('input_tokenizer.pickle', 'rb') as handle:
        input_tokenizer = pickle.load(handle)
        
    with open('output_tokenizer.pickle', 'rb') as handle:
        output_tokenizer = pickle.load(handle)
        
    input_vocab_size = len(input_tokenizer.word_index) + 1
    output_vocab_size = len(output_tokenizer.word_index) + 1
    
    text = preprocess_text(text)  
    seq = input_tokenizer.texts_to_sequences([text])
    inputs = tf.keras.preprocessing.sequence.pad_sequences(seq, truncating='post', padding='post')
    inputs = tf.convert_to_tensor(inputs)
    
    result = ""
    
    encoder = Encoder(input_vocab_size, constants.embedding_dim, constants.units, constants.BATCH_SIZE)
    decoder = Decoder(output_vocab_size, constants.embedding_dim, constants.units, constants.BATCH_SIZE)
    
    checkpoint_dir = './checkpoints'
    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    
    enc_outputs, enc_hidden = encoder(inputs)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([output_tokenizer.word_index['<start>']], 0)
    
    result = beam_search(constants.beam_width, decoder, dec_input, dec_hidden, 
                         enc_outputs, output_tokenizer.word_index['<end>'], output_vocab_size)
    result = output_tokenizer.sequences_to_texts([result])
    print(result)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--translate", dest="input")
    args = parser.parse_args()
    evaluate(args.input)