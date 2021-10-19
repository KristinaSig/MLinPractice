#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature that specifies the number of mentions in each tweet.

Created on Fri Oct 15 16:33:18 2021

@author: KristinaSig
"""

from code.feature_extraction.feature_extractor import FeatureExtractor
import numpy as np
from code.util import COLUMN_MENTIONS

class MentionsCountFeature(FeatureExtractor):
    """Creates a feature that represents the number of hashtags extracted from the tweets."""
    
    # initialize with the input column that includes the extracted mentions
    def __init__(self):
        super().__init__([COLUMN_MENTIONS], COLUMN_MENTIONS + "_count")
            
    
    def _get_values(self, inputs):
 
        # exctract the number of mention ids present               
        counts = np.array(inputs[0].str.count(pat = "'id': "))
        counts = counts.reshape(-1,1)       
        
        return counts