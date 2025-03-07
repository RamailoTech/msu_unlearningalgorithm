#!/bin/bash
# Usage: ./generate_images.sh -m /path/to/model -c /path/to/file.csv

while getopts m:c: flag; do
    case "${flag}" in
        m) MODEL_PATH=${OPTARG};;
        c) CSV_PATH=${OPTARG};;
    esac
done

if [ -z "$MODEL_PATH" ] || [ -z "$CSV_PATH" ]; then
    echo "Usage: $0 -m /path/to/model -c /path/to/file.csv"
    exit 1
fi

echo "Running generate_images.py with:"
echo "Model Path: $MODEL_PATH"
echo "CSV Path:   $CSV_PATH"

python3 scripts/generate_images_for_prompts.py --model_path "$MODEL_PATH" --csv_path "$CSV_PATH"
