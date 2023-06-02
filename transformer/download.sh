#!/bin/bash

# List of files to download, their corresponding MD5 checksums, and target local paths
files_and_checksums=(
  "https://www.dropbox.com/s/4r4dm1hssbdlgb9/models.zip f561c734cad039e3d085790e3d751bd3 models.zip"
  "https://www.dropbox.com/s/8q5cupqw00twvoa/assets.zip c0d56e49fb726d641974c0d211a22a03 assets.zip"
)

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
