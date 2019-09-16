#! /bin/bash

if [ $# -eq 0 ]; then
  echo "$0 NUM_GPU CFG_FILE APP [APP_ARGS]"
  exit 1
fi

#function join { local IFS="$1"; shift; echo "$*"; }

num_gpus=$1
cfg=$2
app=$3
app_args=${@:4}

if [ $num_gpus == 0 ]; then
  echo "$0 NUM_GPU APP [APP_ARGS]"
  echo "Please provide NUM_GPU > 0"
  exit 1
fi

echo "Running $app $app_args on $num_gpus GPUs"

for (( i=1; i<$((num_gpus+1)); ++i ))
do
  echo "Now running on $i gpus"
  sed -i -e "s/gpu=.*/gpu=$i/" $2
  export artsConfig=$2
  $app $app_args
done
