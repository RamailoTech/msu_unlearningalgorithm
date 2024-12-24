#!/bin/bash

# Define URLs for datasets on Hugging Face
SAMPLE_URL="https://huggingface.co/datasets/dipeshlav/sample_i2p_dataset/resolve/main/i2p.csv"
FULL_URL="https://huggingface.co/datasets/your-username/your-dataset-name/resolve/main/full-dataset.zip"

# Define output filenames
SAMPLE_FILE="i2p.csv"
FULL_FILE="full-dataset.zip"

# Define target directories
DATA_DIR="./data/i2p-dataset"
SAMPLE_DIR="$DATA_DIR/sample"
FULL_DIR="$DATA_DIR/full"

# Create directories if they don't exist
prepare_directories() {
  echo "Preparing directories..."
  mkdir -p "$DATA_DIR"
  mkdir -p "$SAMPLE_DIR"
  mkdir -p "$FULL_DIR"
}

# Function to download the dataset
download() {
  local url=$1
  local output=$2
  local target_dir=$3

  echo "Downloading $output from Hugging Face..."
  curl -L -o "$output" "$url"
  if [ $? -eq 0 ]; then
    echo "Download complete: $output"
    mv "$output" "$target_dir"
  else
    echo "Download failed: $output"
  fi
}

# Main logic to handle arguments
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 {sample|full}"
  exit 1
fi

# Parse the argument for sample or full
case "$1" in
  sample)
    URL="$SAMPLE_URL"
    FILE="$SAMPLE_FILE"
    TARGET_DIR="$SAMPLE_DIR"
    ;;
  full)
    URL="$FULL_URL"
    FILE="$FULL_FILE"
    TARGET_DIR="$FULL_DIR"
    ;;
  *)
    echo "Invalid argument. Usage: $0 {sample|full}"
    exit 1
    ;;
esac

# Prepare directories
prepare_directories

# Download the selected dataset
download "$URL" "$FILE" "$TARGET_DIR"
