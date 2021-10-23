#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility file for collecting frequently used constants and helper functions.

Created on Wed Sep 29 10:50:36 2021

@author: lbechberger
"""

# column names for the original data frame
COLUMN_TWEET = "tweet"
COLUMN_LIKES = "likes_count"
COLUMN_RETWEETS = "retweets_count"
COLUMN_HASHTAGS = "hashtags"
COLUMN_MENTIONS = "mentions"
COLUMN_URLS = "urls"
COLUMN_PHOTOS = "photos"
COLUMN_VIDEO = "video"

# column names of novel columns for preprocessing
COLUMN_LABEL = "label"
COLUMN_PUNCTUATION = "tweet_no_punctuation"
COLUMN_TWEET_CLEAN = "tweet_clean"
COLUMN_SENTIMENT = "sentiment_scores"
COLUMN_MEDIA = "contains_media"

SUFFIX_TOKENIZED = "_tokenized"

# attribute options to choose from for the sentiment score
ATTR_NEGATIVE = "neg"
ATTR_POSITIVE = "pos"
ATTR_NEUTRAL = "neu"
ATTR_COMPOUND = "compound"
