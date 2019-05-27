#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 22:29:07 2019

@author: jayasoo
"""

import re
import unicodedata

# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

# Removes unwanted characters from text
def preprocess_text(text):
    text = unicode_to_ascii(text.lower().strip())
    
    # Add space before and after punctuations
    text = re.sub(r"([?.!,])", r" \1 ", text)
    text = re.sub(r'[ ]+', " ", text)
    
    # Remove every character other than what is expected
    text = re.sub(r"[^a-zA-Z0-9?.!,]+", " ", text)
    text = text.rstrip().strip()
    
    # Add start and end tokens
    text = '<start> ' + text + ' <end>'
    
    return text

