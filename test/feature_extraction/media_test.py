#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 20:03:14 2021

@author: KristinaSig
"""

import unittest
import pandas as pd
from code.feature_extraction.media import ContainsMediaFeature
from code.util import COLUMN_URLS, COLUMN_PHOTOS, COLUMN_VIDEO, COLUMN_MEDIA

class MediaFeatureTest(unittest.TestCase):
    
    def setUp(self):
        self.COLUMN_URLS = COLUMN_URLS
        self.COLUMN_PHOTOS = COLUMN_PHOTOS
        self.COLUMN_VIDEO = COLUMN_VIDEO
        self.COLUMN_MEDIA = COLUMN_MEDIA
        self.media_feature = ContainsMediaFeature()
        self.df = pd.DataFrame()
        self.df[self.COLUMN_URLS] = ['[https://blabla]']
        self.df[self.COLUMN_PHOTOS] = ['[]']
        self.df[self.COLUMN_VIDEO] = ['[]']
    
    def test_input_columns(self):
        self.assertEqual(self.media_feature._input_columns, [self.COLUMN_URLS,
                                                             self.COLUMN_PHOTOS,
                                                             self.COLUMN_VIDEO])

    def test_feature_name(self):
        self.assertEqual(self.media_feature.get_feature_name(), COLUMN_MEDIA)

    def test_hashtag_count_correct(self):
        result = self.media_feature.fit_transform(self.df)
        EXPECTED_RESULT = 1
        
        self.assertEqual(result, EXPECTED_RESULT)

if __name__ == '__main__':
    unittest.main()