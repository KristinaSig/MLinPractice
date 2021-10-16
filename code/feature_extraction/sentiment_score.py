#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature to extract the appropriate sentiment score of the tweet.

Created on Sat Oct 16 19:34:31 2021

@author: KristinaSig
"""

# import pandas as pd
import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import COLUMN_SENTIMENT
from code.util import ATTR_COMPOUND as score_attribute

# class for extracting the character-based length as a feature
class SentimentScoreFeature(FeatureExtractor):
    
    # constructor
    def __init__(self):
        # extract the feature values for attribute "compound" or choose a different attribute from util
        super().__init__([COLUMN_SENTIMENT], score_attribute + "_sentiment_score")
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # extract the sentiment score based on the given attribute
    def _get_values(self, inputs):
        """Get a sentiment score for each tweet based on the assigned attribute."""
        
        # # convert the input dictionaries into a dataframe with keys as index
        # scores = pd.DataFrame(inputs[0])
        
        # # extract the column with the desired attribute as sentiment score
        # result = np.array(scores[score_attribute])
        # result = result.reshape(-1,1)
        
        # return result
        
        result = np.array(inputs[0])
        result = result.reshape(-1, 1)
        
        return result


