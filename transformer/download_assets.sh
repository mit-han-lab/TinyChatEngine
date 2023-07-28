#!/bin/bash

# List of files to download, their corresponding MD5 checksums, and target local paths
files_and_checksums=(
  "https://www.dropbox.com/s/8q5cupqw00twvoa/assets.zip 6014d43716e6516a4f7b7161088d3e74 assets.zip"
)

OS=`uname`

# Function to download a file if it doesn't exist or if its MD5 checksum is incorrect
download_if_needed() {
  url="$1"
  expected_md5="$2"
  target_path="$3"

  # Ensure the target directory exists
  target_dir=$(dirname "$target_path")
  mkdir -p "$target_dir"

  # Download the file if it does not exist
  if [ ! -e "$target_path" ]; then
    echo "File '$target_path' does not exist. Downloading..."
    wget -q -O "$target_path" "$url"
  fi

  # Use md5 on MacOS
  if [ $OS = "Darwin" ]
  then
      actual_md5=$(md5 -q "$target_path")
  # Use md5sum on Ubuntu
  elif [ $OS = "Linux" ]
  then
      actual_md5=$(md5sum "$target_path" | cut -d ' ' -f1)
  fi

  if [ "$actual_md5" != "$expected_md5" ]; then
    echo "MD5 checksum for '$target_path' is incorrect. Downloading again..."
    wget -q -O "$target_path" "$url"
  else
    echo "File '$target_path' exists and its MD5 checksum is correct."
  fi
}

# Process each file, its corresponding MD5 checksum, and target local path
for file_and_checksum in "${files_and_checksums[@]}"; do
  url=$(echo "$file_and_checksum" | awk '{ print $1 }')
  expected_md5=$(echo "$file_and_checksum" | awk '{ print $2 }')
  target_path=$(echo "$file_and_checksum" | awk '{ print $3 }')

  download_if_needed "$url" "$expected_md5" "$target_path"
  unzip "$target_path"
done
