#!/bin/bash
env=$1
name=$2
time python scripts/mbexp.py \
    -env $env \
    -logdir log/$env-$name
