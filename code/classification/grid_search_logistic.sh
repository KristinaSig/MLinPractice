#!/bin/bash

mkdir -p data/classification


#Hyperparameter for logistic Regression
solver_val=("liblinear lbfgs sag saga")
c_val=("1 10 100")
class_weight=("balanced None")


# different execution modes
if [ $1 = local ]
then
    echo "[local execution]"
    cmd="code/classification/classifier.sge"
elif [ $1 = grid ]
then
    echo "[grid execution]"
    cmd="qsub code/classification/classifier.sge"
else
    echo "[ERROR! Argument not supported!]"
    exit 1
fi

# do the grid search

for k in $solver_val
do
for j in $c_val
do
for l in $class_weight
do
    echo $k
    echo $j
    echo $l
    $cmd 'data/classification/clf_'"$k"'_'"$j"'_'"$l"'.pickle' --logistic -lr_solver $k -lr_c $j -lr_class_weight $l -s 42 --accuracy --kappa
done
done
done