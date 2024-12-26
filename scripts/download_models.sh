#!/bin/bash

# Function to display usage information
usage() {
  echo "Usage: $0 MODEL_NAME"
  echo "MODEL_NAME should be either 'diffuser' or 'compvis'"
  exit 1
}

# Validate the model name
MODEL="$1"
if [[ "$MODEL" != "diffuser" && "$MODEL" != "compvis" ]]; then
  echo "Invalid model name: $MODEL"
  usage
fi

# Define URLs, output files, and target directories based on the model
if [ "$MODEL" == "diffuser" ]; then
  MODEL_URL="https://huggingface.co/nebulaanish/unlearn_models/resolve/main/diffuser.zip"
  MODEL_FILE="diffuser.zip"
  MODEL_DIR="../models/diffuser"
elif [ "$MODEL" == "compvis" ]; then
  MODEL_URL="https://huggingface.co/nebulaanish/unlearn_models/resolve/main/compvis.zip"
  MODEL_FILE="compvis.zip"
  MODEL_DIR="../models/compvis"
fi

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
