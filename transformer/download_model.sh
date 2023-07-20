#!/bin/bash

# Check if the necessary arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 target_file output_path"
    exit 1
fi

# Define parallel arrays with target files (replace with actual values)
files=("LLaMA_7B_AWQ" "file2")  # ... add more files as needed
urls=("https://www.dropbox.com/s/c00cp1bcrwil8rl/LLaMA_7B_AWQ.zip" "replace_with_your_url2")
checksums=("0d2f13eadb4dd010102dbb473e909743" "replace_with_your_checksum2")

# Get target file
target_file=$1

# Find the target file in the files array and get corresponding URL and checksum
for index in ${!files[@]}; do
   if [ "${files[$index]}" = "${target_file}" ]; then
       source_url=${urls[$index]}
       original_checksum=${checksums[$index]}
       break
   fi
done

# Check if the URL and checksum are found
if [ -z "$source_url" ] || [ -z "$original_checksum" ]; then
    echo "No data found for the target file: $target_file"
    exit 1
fi

# Get output path
output_path=$2

# Create the output directory if it doesn't exist
mkdir -p "$output_path"

file_path="$output_path/$target_file.zip"

# Detect the OS
os=$(uname)

# Compute the checksum based on the OS
if [ "$os" == "Darwin" ]; then
    # MacOS uses md5
    checksum=$(md5 -q "$file_path" 2>/dev/null)
elif [ "$os" == "Linux" ]; then
    # Ubuntu uses md5sum
    checksum=$(md5sum "$file_path" | awk '{ print $1 }' 2>/dev/null)
else
    echo "Unsupported operating system $os. Using md5sum by defualt."
    checksum=$(md5sum "$file_path" | awk '{ print $1 }' 2>/dev/null)
fi

# Compare the computed checksum with the original one
if [ "$checksum" != "$original_checksum" ]; then
    echo "Checksum doesn't match or file does not exist. Downloading..."
    wget -q -O "$file_path" "$source_url" -q --show-progress
    # Unzip the model
    unzip "$file_path" -d "$output_path"
else
    echo "Checksum matches. File is not corrupted."
fi
