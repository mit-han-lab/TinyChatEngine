#!/bin/bash

make clean && make -j

# Find all executable files in the current directory starting with 'profile_'
for file in profile_*; do
  # Check if the file is executable
  if [ -x "$file" ]; then
    echo "Running '$file'..."
    ./"$file"
    exit_code=$?
  fi
done
