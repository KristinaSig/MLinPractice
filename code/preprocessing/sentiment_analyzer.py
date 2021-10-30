#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that evaluates the sentiment polarity and intensity of tweets.

Created on Sat Oct  9 23:33:33 2021

@author: KristinaSig
"""

from code.preprocessing.preprocessor import Preprocessor
from nltk.sentiment import SentimentIntensityAnalyzer

class SentimentAnalyzer(Preprocessor):
    """Assigns sentiment values to input."""
 
    def __init__(self, input_column, output_column):
        """Initialize the Sentiment Analyzer with the given input and output column."""
       
        super().__init__([input_column], output_column)
    
    # don't need to implement _set_variables(), since no variables to set
 
    def _get_values(self, inputs):
        """Get sentiment scores for the tweets."""
        
        analyzer = SentimentIntensityAnalyzer()
        
        # obtain a dictionary of scores for each tweet, containing four scores (neg, neu, pos, compound)
        sentiment = [analyzer.polarity_scores(tweet) for tweet in inputs[0]]
        
        return sentiment
    
    
    