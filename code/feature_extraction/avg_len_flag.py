#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature that captures whether a tweet is longer than average.

Created on Sat Oct 23 13:03:53 2021

@author: saudichya
"""

import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor

class AvgLenFeature(FeatureExtractor):
    """Creates a feature that represents whether the tweet is longer than average based on the given dataset."""

    def __init__(self, input_column):
        """Initialize with a given input column."""
        
        super().__init__([input_column], "{0}_len_flag".format(input_column))
    
    # don't need to fit, so don't overwrite _set_variables()

    def _get_values(self, inputs):
        """Get average length flag."""
        
        result = np.array(inputs[0].str.len())
        result = result.reshape(-1,1)
        
        avg = sum(result)/len(result)
        
        # flag those tweets that are longer than average
        column = np.where(result > avg, 1, 0)
        
        return column

