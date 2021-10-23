#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 19:18:26 2021

@author: KristinaSig
"""

import unittest
import pandas as pd
from code.feature_extraction.hashtags_count import HashtagCountFeature
from code.util import COLUMN_HASHTAGS

class HashtagFeatureTest(unittest.TestCase):
    
    def setUp(self):
        self.COLUMN_HASHTAGS = COLUMN_HASHTAGS
        self.hashtag_feature = HashtagCountFeature()
        self.df = pd.DataFrame()
        self.df[self.COLUMN_HASHTAGS] = ['["one","two"]']
    
    def test_input_columns(self):
        self.assertEqual(self.hashtag_feature._input_columns, [self.COLUMN_HASHTAGS])

    def test_feature_name(self):
        self.assertEqual(self.hashtag_feature.get_feature_name(), self.COLUMN_HASHTAGS + "_count")

    def test_hashtag_count_correct(self):
        result = self.hashtag_feature.fit_transform(self.df)
        EXPECTED_COUNT = 2
        
        self.assertEqual(result, EXPECTED_COUNT)

if __name__ == '__main__':
    unittest.main()