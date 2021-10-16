#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that cleans the raw text of original tweets to only include plain text.

Created on Sat Oct  9 13:03:31 2021

@author: KristinaSig
"""

from code.preprocessing.preprocessor import Preprocessor
import re

class TextCleaner(Preprocessor):
    """Cleans input raw text of hashtags, mentions, urls and unwanted special characters with no meaning."""
    
    def __init__(self, input_column, output_column):
        """Initialize with the given input and output column."""
        super().__init__([input_column], output_column)

    def _get_values(self, inputs):
        
        # identify patterns that should be removed from tweets
        noise = [r"#\S+", r"@\S+", r"https?://\S+", r"[^A-Za-z0-9 _.!?]"]
        
        column = inputs[0]
            
        for pattern in noise:
          
            column = column.str.replace(re.compile(pattern), "")
            
                
        return column






















