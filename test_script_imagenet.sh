#!/bin/bash
set -e

MASTER_ADDR=""
PUBLIC_IP=""
WORLD_SIZE=""
NUM_GPUS=""
RANK=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --master-ip-address) MASTER_ADDR="$2"; shift ;;
        --my-ip-address) PUBLIC_IP="$2"; shift ;;
        --world-size) WORLD_SIZE="$2"; shift ;;
        --num-gpus) NUM_GPUS="$2"; shift ;;
        --rank) RANK="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$MASTER_ADDR" ]  || [ -z "$PUBLIC_IP" ] || [ -z "$WORLD_SIZE" ] || [ -z "$NUM_GPUS" ] || [ -z "$RANK" ]; then
    echo "Error: --ip-address, --world-size, --num-gpus, and --rank are required"
    exit 1
fi

pushd backend/test

./cleanup.sh || true
sleep 1

export MASTER_ADDR
export PUBLIC_IP
export MASTER_PORT=5555
export WORLD_SIZE
export NUM_GPUS
export RANK

export TEST_MODEL=resnet152
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))

export GLOBAL_MISC_COMMAND="--dataset=imagenet100 --num-epochs=100"
export CMDLINE="python test_end_to_end.py --model=$TEST_MODEL $GLOBAL_MISC_COMMAND"

echo "$CMDLINE"
$CMDLINE

popd