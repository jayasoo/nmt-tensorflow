# nmt-tensorflow

# Introduction
This project is an implementation of neural machine translation (NMT) from English to German. Dataset used for training can be found here (http://www.manythings.org/anki/). The model uses seq2seq architecture with an encoder and decoder. Attention mechanism has been added to the decoder to identify which part of the encoder outputs to focus on while translating. Finally, beam search has been implemented for picking the best translation from decoder outputs.

# Usage

# Training
Adjust parameters for the model in constants.py
Start training by running python train.py

# Inference
python evaluate.py -t <Sentence in English>
