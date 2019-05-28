#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 14:25:19 2019

@author: jayasoo
"""

import os
import io
import numpy as np
import tensorflow as tf
import pickle

from models import Encoder, Decoder
import constants
from preprocessing import preprocess_text

tf.enable_eager_execution()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
np.set_printoptions(suppress=True)

def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)

def train_step(inp ,targ, encoder, decoder, optimizer, start_token):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([start_token] * constants.BATCH_SIZE, 1)
        
        # Teacher forcing. Feeding target as next input
        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:,t], predictions)
            dec_input = tf.expand_dims(targ[:,t], 1)
        
    batch_loss = loss / int(targ.shape[1])
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    
    return batch_loss    

def train():
    lines = io.open('data/english_german.txt', encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_text(w) for w in l.split('\t')]  for l in lines]
    english, german = zip(*word_pairs)
    
    input_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    input_tokenizer.fit_on_texts(english)
    input_vocab_size = len(input_tokenizer.word_index) + 1
    
    output_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    output_tokenizer.fit_on_texts(german)
    output_vocab_size = len(output_tokenizer.word_index) + 1
    
    print(input_vocab_size, output_vocab_size)
    print(english[0], german[0])
    print(english[100], german[100])
    
    with open('input_tokenizer.pickle', 'wb') as handle:
        pickle.dump(input_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open('output_tokenizer.pickle', 'wb') as handle:
        pickle.dump(output_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    input_sequences = input_tokenizer.texts_to_sequences(english)
    X_train = tf.keras.preprocessing.sequence.pad_sequences(input_sequences,
                                                            truncating='post', padding='post')
    
    target_sequences = output_tokenizer.texts_to_sequences(german)
    Y_train = tf.keras.preprocessing.sequence.pad_sequences(target_sequences,
                                                            truncating='post', padding='post')
    
    BUFFER_SIZE = len(X_train)
    steps_per_epoch = len(X_train)//constants.BATCH_SIZE
    
    dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(constants.BATCH_SIZE, drop_remainder=True)
    
    encoder = Encoder(input_vocab_size, constants.embedding_dim, constants.units, constants.BATCH_SIZE)
    decoder = Decoder(output_vocab_size, constants.embedding_dim, constants.units, constants.BATCH_SIZE)
    optimizer = tf.train.AdamOptimizer()
    
    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
        
    for epoch in range(constants.EPOCHS):
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, encoder, decoder, optimizer, input_tokenizer.word_index['<start>'])
            total_loss += batch_loss
            if batch % 50 == 0:
                print('Epoch {} batch {} loss {:.4f}'.format(epoch+1, batch, batch_loss))
        if (epoch+1)%2==0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
    

if __name__ == "__main__":
    train()
    