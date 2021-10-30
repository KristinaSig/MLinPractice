#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 19:03:13 2021

@author: KristinaSig
"""

import unittest
import pandas as pd
from code.preprocessing.text_cleaner import TextCleaner
from code.util import COLUMN_TWEET, COLUMN_TWEET_CLEAN

class Text_CleanerTest(unittest.TestCase):
    
    def setUp(self):
        self.COLUMN_TWEET = COLUMN_TWEET
        self.COLUMN_TWEET_CLEAN = COLUMN_TWEET_CLEAN
        self.cleaner = TextCleaner()
        
    def test_input_columns(self):
        self.assertEqual(self.cleaner._input_columns, [self.COLUMN_TWEET])
        
    def test_output_column(self):
        self.assertEqual(self.cleaner._output_column, self.COLUMN_TWEET_CLEAN)
    
    def test_cleaning_single_tweet(self):
        input_tweet = "$%& This is an example tweet with @preprocessor. #funwithunittests http://skgjdajkajdklja;kja;kjda"
        output_tweet = " This is an example tweet with   "
       
        input_df = pd.DataFrame()
        input_df[self.COLUMN_TWEET] = [input_tweet]
        
        cleaned_text = self.cleaner.fit_transform(input_df)
        self.assertEqual(output_tweet, cleaned_text[self.COLUMN_TWEET_CLEAN][0])
    
    def text_cleaning_multiple_tweets(self):
        tweet_1 = "$%& This is an example tweet with @preprocessor."
        tweet_2 = "This is another tweet #funwithunittests http://skgjdajkajdklja;kja;kjda"
        output_tweets = [" This is an example tweet with ", "This is another tweet  "]
        
        input_df = pd.DataFrame()
        input_df[self.COLUMN_TWEET] = [tweet_1, tweet_2]
        
        output_df = pd.DataFrame()
        output_df[self.COLUMN_TWEET_CLEAN] = output_tweets
        
        cleaned_text = self.cleaner.fit_transform(input_df)
        self.assertEqual(output_df, cleaned_text[self.COLUMN_TWEET_CLEAN])       
        
if __name__ == "__main__":
    unittest.main()