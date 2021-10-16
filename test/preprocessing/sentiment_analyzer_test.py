#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 13:36:02 2021

@author: KristinaSig
"""

import unittest
import pandas as pd
import numpy as np
from code.preprocessing.sentiment_analyzer import SentimentAnalyzer

class Sentiment_Test(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.OUTPUT_COLUMN = "output"
        self.sentiment_analyzer = SentimentAnalyzer(self.INPUT_COLUMN, self.OUTPUT_COLUMN)
        
    def test_input_columns(self):
        self.assertEqual(self.sentiment_analyzer._input_columns, [self.INPUT_COLUMN])
        
    def test_output_column(self):
        self.assertEqual(self.sentiment_analyzer._output_column, self.OUTPUT_COLUMN)
    
    def test_sentiment_single_sentence(self):
        input_tweet = "This is the most amazing tweet in history ever."
        output_score = 0.624
       
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_tweet]
        
        sentiment_scores = self.sentiment_analyzer.fit_transform(input_df)
        self.assertEqual(output_score, sentiment_scores[self.OUTPUT_COLUMN][0])

    def test_sentiment_multiple_tweets(self):
        tweet_1 = "This is the most amazing tweet in history ever."
        tweet_2 = "This is the most horrible tweet ever in history, what a disgrace..."
        output_scores = [0.624, -0.5849]
       
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [tweet_1, tweet_2]
        
        output_df = pd.Series()
        output_df[self.OUTPUT_COLUMN] = output_scores
        
        sentiment_scores = self.sentiment_analyzer.fit_transform(input_df)
        np.testing.assert_array_equal(output_scores, sentiment_scores[self.OUTPUT_COLUMN])
        
if __name__ == "__main__":
    unittest.main()