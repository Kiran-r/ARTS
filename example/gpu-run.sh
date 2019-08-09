#! /bin/bash

if [ $# -eq 0 ]; then
  echo "$0 NUM_GPU APP [APP_ARGS]"
  exit 1
fi

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
  echo "Now running on $((i+1)) gpus [${visible_gpus[@]}]"
  CUDA_VISIBLE_DEVICES=${visible_gpus[@]} $app $app_args
done
