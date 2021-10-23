#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 19:46:26 2021

@author: KristinaSig
"""

import unittest
import pandas as pd
from code.feature_extraction.mentions_count import MentionsCountFeature
from code.util import COLUMN_MENTIONS

class HashtagFeatureTest(unittest.TestCase):
    
    def setUp(self):
        self.COLUMN_MENTIONS = COLUMN_MENTIONS
        self.mentions_feature = MentionsCountFeature()
        self.df = pd.DataFrame()
        self.df[self.COLUMN_MENTIONS] = ["[{'screen_name':'stranger1', 'name':'stranger1', 'id':'12345'}, {'screen_name':'stranger2', 'name':'stranger2', 'id':'54321'}]"]
    
    def test_input_columns(self):
        self.assertEqual(self.mentions_feature._input_columns, [self.COLUMN_MENTIONS])

    def test_feature_name(self):
        self.assertEqual(self.mentions_feature.get_feature_name(), self.COLUMN_MENTIONS + "_count")

    def test_hashtag_count_correct(self):
        result = self.mentions_feature.fit_transform(self.df)
        EXPECTED_COUNT = 2
        
        self.assertEqual(result, EXPECTED_COUNT)

if __name__ == '__main__':
    unittest.main()