#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that cleans the raw text of original tweets to only include plain text.

Created on Sat Oct  9 13:03:31 2021

@author: KristinaSig
"""

from code.preprocessing.preprocessor import Preprocessor
from code.util import COLUMN_TWEET, COLUMN_TWEET_CLEAN
import re

class TextCleaner(Preprocessor):
    """Cleans input raw text of hashtags, mentions, urls and unwanted special characters with no meaning."""
    
    def __init__(self):
        """Initialize the Text Cleaner with the already pre-determined input and output column."""
        
        super().__init__([COLUMN_TWEET], COLUMN_TWEET_CLEAN)

    def _set_variables(self, inputs):
        """Store noise patterns that should be removed"""
        
        self._noise = [r"#\S+", r"@\S+", r"https?://\S+", r"[^A-Za-z0-9 _.!?]"]
    
    def _get_values(self, inputs):
        """Clean the input text."""
             
        column = inputs[0]
            
        for pattern in self._noise:
          
            column = column.str.replace(re.compile(pattern), "")
                           
        return column






















