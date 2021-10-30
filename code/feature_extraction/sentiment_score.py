#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature to extract the appropriate sentiment score of the tweet.

Created on Sat Oct 16 19:34:31 2021

@author: KristinaSig
"""

import numpy as np
import pandas as pd
import ast
from code.feature_extraction.feature_extractor import FeatureExtractor
from code.util import COLUMN_SENTIMENT
from code.util import ATTR_COMPOUND as score_attribute

class SentimentScoreFeature(FeatureExtractor):
    """Get the compound sentiment score for each tweet."""
        
    def __init__(self):
        """Initialize with already pre-determined input and output columns."""
        
        super().__init__([COLUMN_SENTIMENT], score_attribute + "_sentiment_score")

    # don't need to fit, so don't overwrite _set_variables()

    def _get_values(self, inputs):
        """Get compound sentiment scores."""
        
        # inputs are treated as strings, so transform into dictionaries first
        list_of_dics = []
        for i in inputs[0]:
            input_dict = ast.literal_eval(i)
            list_of_dics.append(input_dict)
            
        # convert the dictionaries with the scores into a dataframe, keys as index
        scores = pd.DataFrame(list_of_dics)
        
        # extract the column that contains the desired attribute as the final feature
        feature = np.array(scores[score_attribute])
        feature = feature.reshape(-1,1)
        
        return feature


