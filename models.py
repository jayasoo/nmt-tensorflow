#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 20:56:39 2019

@author: jayasoo
"""

import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.encoder_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(self.encoder_units,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer='glorot_uniform'))
        self.fc = tf.keras.layers.Dense(self.encoder_units, activation='relu')
        
    def call(self, x):
        x = self.embedding(x)
        output, state_fw, state_bw = self.gru(x)
        state = self.fc(tf.concat([state_fw, state_bw], axis=-1))
        return output, state

class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
    
    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
    
    
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.CuDNNGRU(self.dec_units,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = Attention(self.dec_units)
        
    def call(self, x, hidden, enc_outputs):
        context_vector, attention_weights = self.attention(hidden, enc_outputs)
        x = self.embedding(x)
        output, state = self.gru(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), output], axis=-1)
        x = tf.reshape(x, (-1,x.shape[-1]))
        x = self.fc(x)
        return x, state, attention_weights