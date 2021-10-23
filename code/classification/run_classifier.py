#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train or evaluate a single classifier with its given set of hyperparameters.

Created on Wed Sep 29 14:23:48 2021

@author: lbechberger
"""

import argparse, pickle
from sklearn.dummy import DummyClassifier

from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import mlflow
from mlflow import log_metric, log_param, set_tracking_uri

# setting up CLI
parser = argparse.ArgumentParser(description = "Classifier")
parser.add_argument("input_file", help = "path to the input pickle file")
parser.add_argument("-s", '--seed', type = int, help = "seed for the random number generator", default = None)
parser.add_argument("-e", "--export_file", help = "export the trained classifier to the given location", default = None)
parser.add_argument("-i", "--import_file", help = "import a trained classifier from the given location", default = None)
parser.add_argument("-m", "--majority", action = "store_true", help = "majority class classifier")
parser.add_argument("-r", "--random", action = "store_true", help = "random uniform classifier")
parser.add_argument("-f", "--frequency", action = "store_true", help = "label frequency classifier")
parser.add_argument("--knn", type = int, help = "k nearest neighbor classifier with the specified value of k", default = None)
parser.add_argument("--random_forest", action = "store_true", help = "random forest classifier")
parser.add_argument("-rf_n_estimators", type = int, help = "the number of trees in a forest", default = 100)
parser.add_argument("-rf_criterion", help = "the function to measure the quality of the split, choose gini or entropy", default = "gini")
parser.add_argument("-rf_depth", type = int, help = "the maximum depth of the tree. When None, the expansion continues as deep as it gets", default = None)
parser.add_argument("-rf_bootstrap", type = bool, help = "determine if bootstrap samples are used for building the trees, if False the whole dataset is used", default = True)
parser.add_argument("-rf_samples", type = int, help = "the number of samples to draw from the set to train each base estimator (relevant only when boostrap is set to True)", default = None)
parser.add_argument("-a", "--accuracy", action = "store_true", help = "evaluate using accuracy")
parser.add_argument("-k", "--kappa", action = "store_true", help = "evaluate using Cohen's kappa")
parser.add_argument("-ap", "--average_precision", action = "store_true", help = "evaluate using average_precision_score")
parser.add_argument("-pr", "--precision_recall_curve", action = "store_true", help = "evaluate using precision_recall_curve")
parser.add_argument("-f1", "--f1_score", action = "store_true", help = "evaluate using F1 score")
parser.add_argument("--log_folder", help = "where to log the mlflow results", default = "data/classification/mlflow")

args = parser.parse_args()

# load data
with open(args.input_file, 'rb') as f_in:
    data = pickle.load(f_in)

set_tracking_uri(args.log_folder)

if args.import_file is not None:
    # import a pre-trained classifier
    with open(args.import_file, 'rb') as f_in:
        input_dict = pickle.load(f_in)
    
    classifier = input_dict["classifier"]
    for param, value in input_dict["params"].items():
        log_param(param, value)
    
    log_param("dataset", "validation")

else:   # manually set up a classifier
    
    if args.majority:
        # majority vote classifier
        print("    majority vote classifier")
        log_param("classifier", "majority")
        params = {"classifier": "majority"}
        classifier = DummyClassifier(strategy = "most_frequent", random_state = args.seed)
        
    elif args.frequency:
        # label frequency classifier
        print("    label frequency classifier")
        log_param("classifier", "frequency")
        params = {"classifier": "frequency"}
        classifier = DummyClassifier(strategy = "stratified", random_state = args.seed)
    
    elif args.random:
        #random uniform classifier
        print("    random uniform classifier")
        classifier = DummyClassifier(strategy = "uniform", random_state = args.seed)
        	        
    elif args.knn is not None:
        print("    {0} nearest neighbor classifier".format(args.knn))
        log_param("classifier", "knn")
        log_param("k", args.knn)
        params = {"classifier": "knn", "k": args.knn}
        standardizer = StandardScaler()
        knn_classifier = KNeighborsClassifier(args.knn, n_jobs = -1)
        classifier = make_pipeline(standardizer, knn_classifier)
    
    elif args.random_forest:
        print("    random forest classifier")
        log_param("classifier", "random forest")
        log_param("n_estimators", args.rf_n_estimators)
        log_param("criterion", args.rf_criterion)
        log_param("max_depth", args.rf_depth)
        log_param("bootstrap", args.rf_bootstrap)
        log_param("max_samples", args.rf_samples)
        params = {"classifier": "random forest", 
                  "n_estimators":args.rf_n_estimators, 
                  "criterion": args.rf_criterion,
                  "max_depth": args.rf_depth,
                  "bootstrap": args.rf_bootstrap,
                  "max_samples": args.rf_samples}
        
        n_est = args.rf_n_estimators
        crit = args.rf_criterion
        depth = args.rf_depth
        bs = args.rf_bootstrap
        samples = args.rf_samples
        classifier = RandomForestClassifier(n_estimators = n_est, 
                                            criterion = crit, 
                                            max_depth = depth, 
                                            bootstrap = bs, 
                                            max_samples = samples, 
                                            n_jobs = -1)
    
    classifier.fit(data["features"], data["labels"].ravel())
    log_param("dataset", "training")

# now classify the given data
prediction = classifier.predict(data["features"])

# collect all evaluation metrics
evaluation_metrics = []
if args.accuracy:
    evaluation_metrics.append(("accuracy", accuracy_score))
if args.kappa:
    evaluation_metrics.append(("Cohen_kappa", cohen_kappa_score))
if args.average_precision:
    evaluation_metrics.append(("Average_precision_score", average_precision_score))
if args.precision_recall_curve:
    evaluation_metrics.append(("Precision_Recall_Curve", precision_recall_curve))
if args.f1_score:
    evaluation_metrics.append(("F1 score", f1_score))

# compute and print them
for metric_name, metric in evaluation_metrics:
    metric_value = metric(data["labels"], prediction)
    print("    {0}: {1}".format(metric_name, metric_value))
    log_metric(metric_name, metric_value)
    
# export the trained classifier if the user wants us to do so
if args.export_file is not None:
    output_dict = {"classifier": classifier, "params": params}
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(output_dict, f_out)
