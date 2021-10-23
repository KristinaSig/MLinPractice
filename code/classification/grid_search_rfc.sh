#!/bin/bash

mkdir -p data/classification

# specify hyperparameter values
values_n=("50 100 200 300 400 500")
values_criterion=("gini entropy")
values_depth=("10 50 100")
values_bootstrap=("True False")
values_samples=("1000 5000 10000")

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
for n in $values_n
do for criterion in $values_criterion
do for depth in $values_depth
do for bs in $values_bootstrap
do for samples in $values_samples
do
    echo $n $criterion $depth $bs $samples
    $cmd 'data/classification/clf_'"$n"'_'"$criterion"'_'"$depth"'_'"$bs"'_'"$samples"'.pickle' --random_forest -rf_n_estimators $n -rf_criterion $criterion -rf_depth $depth -rf_bootstrap $bs -rf_samples $samples -s 42 --accuracy --kappa --f1_score
done
done
done
done
done