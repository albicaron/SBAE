#!/bin/bash

NUM_CORES=15

for ((i=1; i<=$NUM_CORES; i++))
do
  SEED=$i
  OUTPUT_FILE="MountCar${i}"
  # MAX_MODEL_RUNS=1
  # MAX_TRAINING_EPISODES=30000
  # Record the start time
  start_time=$(date +%s)

  nohup python3 -u 1_main_mountaincar.py --seed $SEED > "nohup/nohup_vm/${OUTPUT_FILE}.log" 2>&1 &
  disown

  # Wait for the process to complete
  wait
  # Record the end time
  end_time=$(date +%s)

  # Calculate the time difference
  elapsed_time=$((end_time - start_time))

  # Append the elapsed time to the log file
  echo "Iteration ${i} took ${elapsed_time} seconds" >> "nohup/nohup_vm/${OUTPUT_FILE}.log"
done

