#!/bin/bash

make clean && make -j

# Find all executable files in the current directory starting with 'test_'
for file in test_*; do
  # Check if the file is executable
  if [ -x "$file" ]; then
    echo "Running '$file'..."
    ./"$file"
    exit_code=$?
  fi
done
