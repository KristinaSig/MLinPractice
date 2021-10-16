#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 19:03:13 2021

@author: KristinaSig
"""

import unittest
import pandas as pd
from code.preprocessing.text_cleaner import TextCleaner

class Text_CleanerTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.OUTPUT_COLUMN = "output"
        self.cleaner = TextCleaner(self.INPUT_COLUMN, self.OUTPUT_COLUMN)
        
    def test_boolean(self):
        self.assertEqual(True, not False)
        
    def test_input_columns(self):
        self.assertEqual(self.cleaner._input_columns, [self.INPUT_COLUMN])
        
    def test_output_column(self):
        self.assertEqual(self.cleaner._output_column, self.OUTPUT_COLUMN)
    
    def test_cleaning_single_tweet(self):
        input_tweet = "$%& This is an example tweet with @preprocessor. #funwithunittests http://skgjdajkajdklja;kja;kjda"
        output_tweet = " This is an example tweet with   "
       
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_tweet]
        
        cleaned_text = self.cleaner.fit_transform(input_df)
        self.assertEqual(output_tweet, cleaned_text[self.OUTPUT_COLUMN][0])
    
    def text_cleaning_multiple_tweets(self):
        tweet_1 = "$%& This is an example tweet with @preprocessor."
        tweet_2 = "This is another tweet #funwithunittests http://skgjdajkajdklja;kja;kjda"
        output_tweets = [" This is an example tweet with ", "This is another tweet  "]
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [tweet_1, tweet_2]
        
        output_df = pd.DataFrame()
        output_df[self.OUTPUT_COLUMN] = output_tweets
        
        cleaned_text = self.cleaner.fit_transform(input_df)
        self.assertEqual(output_df, cleaned_text[self.OUTPUT_COLUMN])       
        
if __name__ == "__main__":
    unittest.main()