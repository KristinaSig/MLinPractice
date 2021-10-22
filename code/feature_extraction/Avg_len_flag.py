#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 00:02:44 2021

@author: saudichya
"""

import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor

# class for extracting Binary flag on the basis of Average tweet length
class AvgLenTweet(FeatureExtractor):
    
    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_Len_Flag".format(input_column))
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # compute the word length based on the inputs
    def _get_values(self, inputs):
        
        result = np.array(inputs[0].str.len())
        result = result.reshape(-1,1)
        avg = sum(result)/len(result)
        result1 = np.where(result > avg, 1, 0)
        return result1









