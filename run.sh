#!/usr/bin/env sh

############### Host   ##############################
HOST=$(hostname)
echo "Current host is: $HOST"


# path to benchmark
benchmark="$PYTORX_HOME/benchmark/mnist.py"


############### Neural network ############################
epochs=20
batch_size=1000
test_batch_size=100
crxb_size=64
vdd=3.3
gwire=0.0375
gload=0.25
gmax=0.000333
gmin=0.000000333


{
$PYTHON $benchmark  --epochs $epochs \
                    --batch_size $batch_size\
                    --test_batch_size $test_batch_size\
                    --crxb_size $crxb_size\
                    --vdd $vdd\
                    --gwire $gwire\
                    --gload $gload\
                    --gmax $gmax\
                    --gmin $gmin
} &

