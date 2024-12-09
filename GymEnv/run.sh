#!/bin/bash

# Define arrays for the parameters
envs=("Taxi-v3" "TaxiUncertainty-v0")
lrs=(0.0003 0.0007)
seeds=(10 20 30 40 50 60 70 80 90 100)

# Iterate over the parameter combinations
for env in "${envs[@]}"; do
  for lr in "${lrs[@]}"; do
    for seed in "${seeds[@]}"; do
      echo "Running main.py with --env=$env, --lr=$lr, and --seed=$seed"
      python main.py --env "$env" --lr "$lr" --seed "$seed"
    done
  done
done
