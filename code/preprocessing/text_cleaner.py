#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that cleans the raw text of original tweets to only include plain text.

Created on Sat Oct  9 13:03:31 2021

@author: KristinaSig
"""

from code.preprocessing.preprocessor import Preprocessor
import re

class Text_cleaner(Preprocessor):
    """Cleans input raw text of hashtags, mentions, urls and unwanted special characters with no meaning."""
    
    def __init__(self, input_column, output_column):
        #input column "tweet", output column new with clean data
        super().__init__([input_column], output_column)

    
    def _get_values(self, inputs):
        
        noise = [r"#\S+", r"@\S+", r"https?://\S+", r"[^A-Za-z0-9 _.!?]"]
        
        clean_tweets = []
        
        for tweet in inputs[0]:
            
            text_to_clean = tweet
            
            for pattern in noise:
          
                text_to_clean = re.sub(pattern,"", text_to_clean)
            
            clean_tweets.append(str(text_to_clean))
                
        return clean_tweets