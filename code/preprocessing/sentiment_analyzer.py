#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that assigns a positive, negative or neutral value to a tweet.

Created on Sat Oct  9 23:33:33 2021

@author: KristinaSig
"""

from code.preprocessing.preprocessor import Preprocessor
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np

class SentimentAnalyzer(Preprocessor):
    """Assigns sentiment value to input."""
 
    def __init__(self, input_column, output_column):
        super().__init__([input_column], output_column)
 
    def _get_values(self, inputs):
        
        analyzer = SentimentIntensityAnalyzer()
        
        # obtain a compound score, which is a balanced version of the negative-neutral-positive scores
        sentiment = [analyzer.polarity_scores(tweet)["compound"] for tweet in inputs[0]]
        
        sentiment_scores = np.array(sentiment)
        sentiment_scores = sentiment_scores.reshape(-1,1)
        
        return sentiment_scores
    
    
    