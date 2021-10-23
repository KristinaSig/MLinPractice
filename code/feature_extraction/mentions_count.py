#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature that specifies the number of mentions in each tweet.

Created on Fri Oct 15 16:33:18 2021

@author: KristinaSig
"""

from code.feature_extraction.feature_extractor import FeatureExtractor
import numpy as np
import ast
from code.util import COLUMN_MENTIONS

class MentionsCountFeature(FeatureExtractor):
    """Creates a feature that represents the number of mentions extracted from the tweets."""
    
    # initialize with the input column that includes the extracted mentions
    def __init__(self):
        super().__init__([COLUMN_MENTIONS], COLUMN_MENTIONS + "_count")
            
    
    def _get_values(self, inputs):
 
        counts = []
        for i in inputs[0]:
            # first, transform each string input into a list
            input_list = ast.literal_eval(i)
            
            mentions_count = len(input_list)
            counts.append(mentions_count)
        
        column = np.array(counts)
        column = column.reshape(-1,1)     
        
        return column