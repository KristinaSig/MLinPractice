#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature that captures whether the tweet contains any media.

Created on Fri Oct 15 17:15:18 2021

@author: KristinaSig
"""

from code.feature_extraction.feature_extractor import FeatureExtractor
import pandas as pd
import numpy as np
from code.util import COLUMN_URLS, COLUMN_PHOTOS, COLUMN_VIDEO, COLUMN_MEDIA

class ContainsMediaFeature(FeatureExtractor):
    """Creates a feature that indicates presence of some media identified in the tweet."""
    
    def __init__(self):
        """Initialize with already pre-determined input and output columns."""
        
        super().__init__([COLUMN_URLS, COLUMN_PHOTOS, COLUMN_VIDEO], COLUMN_MEDIA)
            
    # don't need to fit, so don't overwrite _set_variables()
  
    def _get_values(self, inputs):
        """Get media flag."""
        
        # if returned True, the input list contains some urls  
        tweet_urls = [not i for i in inputs[0].str.contains(pat = "\[]")]
        
        # if returned True, the input list contains some photos
        tweet_photos = [not i for i in inputs[1].str.contains(pat = "\[]")]
        
        # if 1, the input indicates presence of a video in the tweet
        tweet_video = inputs[2]
        
        # add the three lists into a single dataframe
        data = pd.DataFrame(list(zip(tweet_urls, tweet_photos, tweet_video)))
        
        # by summing the values, we identify if one or more of the media types is True, ie. any media is identified
        has_media = data[list(data.columns)].sum(axis = 1)
        has_media = [True if i>0 else False for i in has_media]
        
        result = np.array(has_media)
        result = result.reshape(-1,1)
        
        return result