#!/bin/bash

# Check if the necessary arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 target_file output_path"
    exit 1
fi

# Define dictionary with target files (replace with actual values)
declare -A files
files["LLaMA_7B_AWQ"]="https://www.dropbox.com/s/c00cp1bcrwil8rl/LLaMA_7B_AWQ.zip 0d2f13eadb4dd010102dbb473e909743"
# ... add more files as needed

# Get target file, URL, and checksum
target_file="$1.zip"
info=(${files[$target_file]})
source_url=${info[0]}
original_checksum=${info[1]}

# Check if the URL and checksum are found
if [ -z "$source_url" ] || [ -z "$original_checksum" ]; then
    echo "No data found for the target file: $target_file"
    exit 1
fi

# Get output path
output_path=$2

# Create the output directory if it doesn't exist
mkdir -p "$output_path"
file_path="$output_path/$target_file"

# Detect the OS
os=$(uname)

# Compute the checksum based on the OS
if [ "$os" == "Darwin" ]; then
    # MacOS uses md5
    checksum=$(md5 -q "$file_path")
elif [ "$os" == "Linux" ]; then
    # Ubuntu uses md5sum
    checksum=$(md5sum "$file_path" | awk '{ print $1 }')
else
    echo "Unsupported operating system."
    exit 1
fi

# Compare the computed checksum with the original one
if [ "$checksum" != "$original_checksum" ]; then
    echo "Checksum doesn't match. File is corrupted. Downloading a new copy..."
    curl -o "$file_path" "$source_url"
    # Unzip the model
    unzip file_path -d output_path
else
    echo "Checksum matches. File is not corrupted."
fi
