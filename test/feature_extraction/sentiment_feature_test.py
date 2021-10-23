#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 18:39:31 2021

@author: KristinaSig
"""

import unittest
import pandas as pd
from code.feature_extraction.sentiment_score import SentimentScoreFeature
from code.util import ATTR_COMPOUND as score_attribute
from code.util import COLUMN_SENTIMENT

class SentimentFeatureTest(unittest.TestCase):
    
    def setUp(self):
        self.COLUMN_SENTIMENT = COLUMN_SENTIMENT
        self.sentiment_feature = SentimentScoreFeature()
        self.df = pd.DataFrame()
        self.df[self.COLUMN_SENTIMENT] = ["{'neg': 0.0, 'neu': 0.662, 'pos': 0.338, 'compound': 0.624}"] 
    
    def test_input_columns(self):
        self.assertEqual(self.sentiment_feature._input_columns, [self.COLUMN_SENTIMENT])

    def test_feature_name(self):
        self.assertEqual(self.sentiment_feature.get_feature_name(), score_attribute + "_sentiment_score")

    def test_sentiment_score_correct(self):
        result = self.sentiment_feature.fit_transform(self.df)
        EXPECTED_SCORE = 0.624
                
        self.assertEqual(result, EXPECTED_SCORE)


if __name__ == '__main__':
    unittest.main()