#!/usr/bin/env sh

############### Host   ##############################
HOST=$(hostname)
echo "Current host is: $HOST"

# path to python exec
PYTHON="/home/ubuntu/anaconda3/envs/pytorch1p1/bin/python"

# path to benchmark
benchmark="/home/ubuntu/PytorX/benchmark/mnist.py"


############### Neural network ############################
epochs=20


{
$PYTHON $benchmark --epochs $epochs \

} &

