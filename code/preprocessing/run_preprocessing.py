#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the specified collection of preprocessing steps

Created on Tue Sep 28 16:43:18 2021

@author: lbechberger
"""

import argparse, csv, pickle
import pandas as pd
from sklearn.pipeline import make_pipeline
from code.preprocessing.punctuation_remover import PunctuationRemover
from code.preprocessing.text_cleaner import TextCleaner
from code.preprocessing.tokenizer import Tokenizer
from code.preprocessing.sentiment_analyzer import SentimentAnalyzer
from code.util import COLUMN_TWEET_CLEAN, SUFFIX_TOKENIZED, COLUMN_SENTIMENT

# setting up CLI
parser = argparse.ArgumentParser(description = "Various preprocessing steps")
parser.add_argument("input_file", help = "path to the input csv file")
parser.add_argument("output_file", help = "path to the output csv file")
parser.add_argument("-p", "--punctuation", action = "store_true", help = "remove punctuation")
parser.add_argument("-c", "--clean_text", action = "store_true", help = "clean text of linguistically non-relevant parts, such as hashtags, mentions, urls")
parser.add_argument("-t", "--tokenize", action = "store_true", help = "tokenize given column into individual words")
parser.add_argument("-s", "--analyze_sentiment", action = "store_true", help = "assign a sentiment score to each tweet in a column")
parser.add_argument("--tokenize_input", help = "input column to tokenize", default = COLUMN_TWEET_CLEAN)
parser.add_argument("--sentiment_input", help = "input column to sentiment analyzer", default = COLUMN_TWEET_CLEAN)
parser.add_argument("-e", "--export_file", help = "create a pipeline and export to the given location", default = None)
args = parser.parse_args()

# load data
df = pd.read_csv(args.input_file, quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")

# collect all preprocessors
preprocessors = []
if args.punctuation:
    preprocessors.append(PunctuationRemover())
if args.clean_text:
    preprocessors.append(TextCleaner())
if args.tokenize:
    preprocessors.append(Tokenizer(args.tokenize_input, args.tokenize_input + SUFFIX_TOKENIZED))
if args.analyze_sentiment:
    preprocessors.append(SentimentAnalyzer(args.sentiment_input, COLUMN_SENTIMENT))

# call all preprocessing steps
for preprocessor in preprocessors:
    df = preprocessor.fit_transform(df)

# store the results
df.to_csv(args.output_file, index = False, quoting = csv.QUOTE_NONNUMERIC, line_terminator = "\n")

# create a pipeline if necessary and store it as pickle file
if args.export_file is not None:
    pipeline = make_pipeline(*preprocessors)
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(pipeline, f_out)