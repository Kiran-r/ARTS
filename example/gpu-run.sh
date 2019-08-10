#! /bin/bash

if [ $# -eq 0 ]; then
  echo "$0 NUM_GPU APP [APP_ARGS]"
  exit 1
fi

function join { local IFS="$1"; shift; echo "$*"; }

num_gpus=$1
app=$2
visible_gpus=()
app_args=${@:3}

if [ $num_gpus == 0 ]; then
  echo "$0 NUM_GPU APP [APP_ARGS]"
  echo "Please provide NUM_GPU > 0"
  exit 1
fi

echo "Running $app $app_args on $num_gpus GPUs"

for (( i=0; i<$num_gpus; ++i ))
do
  visible_gpus+=($i)
  gpus=$(join , ${visible_gpus[@]})
  echo "Now running on $((i+1)) gpus $gpus"
  CUDA_VISIBLE_DEVICES=$gpus $app $app_args
done
