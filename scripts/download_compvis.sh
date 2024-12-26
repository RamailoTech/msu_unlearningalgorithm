#!/bin/bash

# Define URL for the compvis model
MODEL_URL="https://huggingface.co/nebulaanish/unlearn_models/resolve/main/compvis.zip"

# Define output filename
MODEL_FILE="compvis.zip"

# Define target directory (one level up from the current directory)
MODEL_DIR="../models/compvis"

# Create directories if they don't exist
prepare_directories() {
  echo "Preparing directories..."
  mkdir -p "$MODEL_DIR"
}

# Function to download and extract the model
download_and_extract() {
  local url=$1
  local output=$2
  local target_dir=$3

  echo "Downloading $output from $url..."
  curl -L -o "$output" "$url"
  if [ $? -eq 0 ]; then
    echo "Download complete: $output"
    echo "Testing if $output is a valid ZIP file..."
    if unzip -t "$output" >/dev/null 2>&1; then
      echo "Extracting $output to $target_dir..."
      unzip -o "$output" -d "$target_dir"
      if [ $? -eq 0 ]; then
        echo "Extraction complete: $target_dir"
        rm "$output"  # Clean up the downloaded ZIP file
      else
        echo "Extraction failed."
      fi
    else
      echo "$output is not a valid ZIP file."
    fi
  else
    echo "Download failed: $output"
  fi
}

# Main logic
prepare_directories
download_and_extract "$MODEL_URL" "$MODEL_FILE" "$MODEL_DIR"
