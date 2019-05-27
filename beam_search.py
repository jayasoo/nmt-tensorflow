#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 14:47:03 2019

@author: jayasoo
"""

import numpy as np
import tensorflow as tf
import constants

def length_of_beam(beam_result, end_token):
    for i, token in enumerate(beam_result):
        if token == end_token:
            return i+1
    return len(beam_result)

def beam_step(beam_width, step, decoder, dec_input, dec_hidden, enc_outputs, 
              end_token, vocab_size, score):
    # First beam step
    if step == 1:
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_outputs)
        predictions = np.exp(predictions[0])/sum(np.exp(predictions[0]))
        predictions = tf.log(predictions)
        values, indices = tf.math.top_k(predictions, k=beam_width)
        dec_input_t = tf.concat([tf.broadcast_to(dec_input, [beam_width,1]), tf.expand_dims(indices,1)], axis=1)
        dec_hidden_t = tf.broadcast_to(dec_hidden, [beam_width,dec_hidden.shape[1]])
        score_t = values
        return dec_input_t, dec_hidden_t, score_t
    
    beams_ended = []
    
    # Generate next set of predictions from the previous beams
    predictions, dec_hidden, attention_weights = decoder(tf.expand_dims(dec_input[:,step-1],1), dec_hidden, enc_outputs)
    scores = predictions.numpy()
    scores_length_normalized = np.zeros(scores.shape)
    for b in range(beam_width):
        if dec_input[b,step-1] == end_token:
            # Beam has already reached end token
            beams_ended.append(b)
            # Set first value as the previous beam score and rest as -ve inf 
            # for all the tokens generated from the ended beam
            scores[b] = np.append([score[b]], [-np.inf] * (vocab_size-1))
        else:
            # Beam hasn't reached end token
            scores[b] = np.exp(scores[b])/sum(np.exp(scores[b]))
            scores[b] = tf.log(scores[b])
            scores[b] = scores[b] + score[b]
        # Length normalization of beam
        scores_length_normalized[b] = scores[b] / np.power(length_of_beam(dec_input[b],end_token), 0.7)
     
    # Initializing next set of inputs for beam step    
    dec_input_t = np.zeros([beam_width, step+1])
    dec_hidden_t = dec_hidden
    score_t = np.zeros([beam_width])
    
    # Unrolling scores    
    scores = np.reshape(scores, (1,-1))
    scores_length_normalized = np.reshape(scores_length_normalized, (1,-1))
    values, indices = tf.math.top_k(scores_length_normalized[0], k=beam_width)
    
    for i, index in enumerate(indices):
        beam = index//vocab_size
        if beam.numpy() in beams_ended:
            # If the beam has already ended append end token
            dec_input_t[i] = np.append(dec_input[beam], [end_token])
        else:
            # If beam hasn't ended add the next token which gives higher score
            dec_input_t[i,:] = np.append(dec_input[beam,:], [index%vocab_size])
        score_t[i] = scores[0][index]
    
    return dec_input_t, dec_hidden_t, score_t


def beam_search(beam_width, decoder, dec_input, dec_hidden, enc_outputs, end_token, vocab_size):
    score = None
    for t in range(1,constants.beam_length):
        dec_input, dec_hidden, score = beam_step(beam_width, t, decoder, dec_input, 
                                                 dec_hidden, enc_outputs, end_token, vocab_size, score)
    
    best_beam = tf.argmax(score)
    output = []
    for token in dec_input[best_beam]:
        output.append(token)
        if token == end_token:
            break
    return output