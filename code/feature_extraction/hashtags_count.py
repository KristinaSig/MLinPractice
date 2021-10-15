#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature that specifies the number of hashtags in each tweet.

Created on Thu Oct 14 18:42:58 2021

@author: KristinaSig
"""

from code.feature_extraction.feature_extractor import FeatureExtractor
import numpy as np
from code.util import COLUMN_HASHTAGS

class HashtagCountFeature(FeatureExtractor):
    """Creates a feature that represents the number of hashtags extracted from the tweets."""
    
    # initialize with the input column that includes the extracted hashtags
    def __init__(self):
        super().__init__([COLUMN_HASHTAGS], COLUMN_HASHTAGS + "_count")
            
    
    def _get_values(self, inputs):
               
        # counts = inputs[0].str.len()
        counts = np.array(inputs[0].str.len())
        counts = counts.reshape(-1,1)
        
        return counts