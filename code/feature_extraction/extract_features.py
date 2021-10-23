#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the specified collection of feature extractors.

Created on Wed Sep 29 11:00:24 2021

@author: lbechberger
"""

import argparse, csv, pickle
import pandas as pd
import numpy as np
from code.feature_extraction.character_length import CharacterLength
from code.feature_extraction.Avg_len_tweet_flag import AvgLenTweet
from code.feature_extraction.hashtags_count import HashtagCountFeature
from code.feature_extraction.mentions_count import MentionsCountFeature
from code.feature_extraction.media import ContainsMediaFeature
from code.feature_extraction.feature_collector import FeatureCollector
from code.util import COLUMN_TWEET, COLUMN_LABEL


# setting up CLI
parser = argparse.ArgumentParser(description = "Feature Extraction")
parser.add_argument("input_file", help = "path to the input csv file")
parser.add_argument("output_file", help = "path to the output pickle file")
parser.add_argument("-e", "--export_file", help = "create a pipeline and export to the given location", default = None)
parser.add_argument("-i", "--import_file", help = "import an existing pipeline from the given location", default = None)
parser.add_argument("-c", "--char_length", action = "store_true", help = "compute the number of characters in the tweet")
parser.add_argument("-avg", "--avg_char_len_flag", action = "store_true", help = "compute the binary flag on the basis of char length")
parser.add_argument("-hc", "--hashtag_count", action = "store_true", help = "count the number of hashtags extracted from the tweet")
parser.add_argument("-mc", "--mentions_count", action = "store_true", help = "count the number of mentions extracted from the tweet")
parser.add_argument("-m", "--media", action = "store_true", help = "state whether there was any media found in the tweet")
args = parser.parse_args()

# load data
df = pd.read_csv(args.input_file, quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")

if args.import_file is not None:
    # simply import an exisiting FeatureCollector
    with open(args.import_file, "rb") as f_in:
        feature_collector = pickle.load(f_in)

else:    # need to create FeatureCollector manually

    # collect all feature extractors
    features = []
    if args.char_length:
        # character length of original tweet (without any changes)
        features.append(CharacterLength(COLUMN_TWEET))
    if args.hashtag_count:
        # count of hashtags extracted in the hashtags column
        features.append(HashtagCountFeature())
    if args.mentions_count:
        # count of mentions extracted in the hashtags column
        features.append(MentionsCountFeature())
    if args.media:
        # state the presence of any media in the tweet
        features.append(ContainsMediaFeature())
    if args.avg_char_len_flag:
        # character length of original tweet (without any changes)
        features.append(AvgLenTweet(COLUMN_TWEET))
    
    # create overall FeatureCollector
    feature_collector = FeatureCollector(features)
    
    # fit it on the given data set (assumed to be training data)
    feature_collector.fit(df)


# apply the given FeatureCollector on the current data set
# maps the pandas DataFrame to an numpy array
feature_array = feature_collector.transform(df)

# get label array
label_array = np.array(df[COLUMN_LABEL])
label_array = label_array.reshape(-1, 1)

# store the results
results = {"features": feature_array, "labels": label_array, 
           "feature_names": feature_collector.get_feature_names()}
with open(args.output_file, 'wb') as f_out:
    pickle.dump(results, f_out)

# export the FeatureCollector as pickle file if desired by user
if args.export_file is not None:
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(feature_collector, f_out)