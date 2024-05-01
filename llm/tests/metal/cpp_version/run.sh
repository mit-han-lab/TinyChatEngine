#!/bin/bash

# Initialize sum
sum=0

# Number of runs
runs=5

for i in $(seq 1 $runs); do
    # Run the program and capture its output
    output=$(./main)
    echo "Run $i: $output"
    # Add the output to sum
    sum=$(echo "$sum + $output" | bc)
done

# Calculate the average
average=$(echo "scale=2; $sum / $runs" | bc)

echo "Total sum: $sum"
echo "Average: $average"