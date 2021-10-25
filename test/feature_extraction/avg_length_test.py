#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 01:17:10 2021

@author: KristinaSig
"""

import unittest
import pandas as pd
import numpy as np
from code.feature_extraction.avg_len_flag import AvgLenFeature

class AvgLenTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.avg_len_feature = AvgLenFeature(self.INPUT_COLUMN)
        self.df = pd.DataFrame()
        self.df[self.INPUT_COLUMN] = ["Hello, this is the first tweet.", "This is also one.", "And this."]
        
    def test_input_columns(self):
        self.assertEqual(self.avg_len_feature._input_columns, [self.INPUT_COLUMN])

    def test_feature_name(self):
        self.assertEqual(self.avg_len_feature.get_feature_name(), "{0}_len_flag".format(self.INPUT_COLUMN))

    def test_avg_length_flag_correct(self):
        result = self.avg_len_feature.fit_transform(self.df)
        EXPECTED_RESULT = [[1],[0],[0]]
        expected_result = np.array(EXPECTED_RESULT)
        
        np.testing.assert_array_equal(result, expected_result)

if __name__ == '__main__':
    unittest.main()