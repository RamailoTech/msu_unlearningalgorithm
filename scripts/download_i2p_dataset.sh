#!/bin/bash

# Check if the required argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <full|sample>"
    exit 1
fi

# Dataset type
DATASET_TYPE=$1

# Repository and URLs
FULL_REPO="https://huggingface.co/datasets/nebulaanish/i2p-dataset"
SAMPLE_ZIP_URL="https://huggingface.co/datasets/dipeshlav/sample_i2p_dataset/resolve/main/sample.zip"

# Output directories
BASE_DIR="data/i2p-dataset"
FULL_DIR="data/i2p-dataset"
SAMPLE_DIR="data/i2p-dataset"


# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for required commands
check_requirements() {
    local missing_deps=()
    
    if ! command_exists git; then
        missing_deps+=("git")
    fi
    
    if ! command_exists git-lfs; then
        missing_deps+=("git-lfs")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        echo "Error: Missing required dependencies: ${missing_deps[*]}"
        echo "Please install them before continuing."
        echo "For Ubuntu/Debian: sudo apt-get install ${missing_deps[*]}"
        echo "For MacOS: brew install ${missing_deps[*]}"
        exit 1
    fi
}

check_existing_directory() {
    if [ "$DATASET_TYPE" == "full" ] && [ -d "$FULL_DIR/full" ]; then
        echo "Error: Directory '$FULL_DIR/full'  already exists. Please remove it before proceeding."
        exit 1
    fi
    
    if [ "$DATASET_TYPE" == "sample" ] && [ -d "$SAMPLE_DIR/sample" ]; then
        echo "Error: Directory '$SAMPLE_DIR/sample' already exists. Please remove it before proceeding."
        exit 1
    fi
}

# Download full dataset using Git LFS
download_full_dataset() {
    echo "Downloading full dataset..."
    
    mkdir -p "$FULL_DIR"
    cd "$FULL_DIR" || exit 1
    
    git lfs install
    
    # Clone the repository with Git LFS
    mkdir -p "temp" && cd "temp" 
    if git clone "$FULL_REPO" .; then
        echo "Full dataset downloaded successfully to $FULL_DIR"
        mv * .. && cd .. && rm -rf "temp"
    else
        echo "Error: Failed to download the full dataset"
        exit 1
    fi

}



# Download and extract sample dataset
download_sample_dataset() {
    echo "Downloading sample dataset..."
    mkdir -p "$SAMPLE_DIR"
    
    local zip_file="$SAMPLE_DIR/sample.zip"
    
    if wget -q --show-progress -O "$zip_file" "$SAMPLE_ZIP_URL"; then
        echo "Extracting sample dataset..."
        if unzip -q "$zip_file" -d "$SAMPLE_DIR"; then
            rm "$zip_file"
            echo "Sample dataset extracted to $SAMPLE_DIR"
        else
            echo "Error: Failed to extract the sample dataset"
            rm "$zip_file"
            exit 1
        fi
    else
        echo "Error: Failed to download the sample dataset"
        exit 1
    fi
}

# Main execution
check_existing_directory
case "$DATASET_TYPE" in
    "full")
        check_requirements
        download_full_dataset
        ;;
    "sample")
        download_sample_dataset
        ;;
    *)
        echo "Invalid dataset type. Use 'full' or 'sample'."
        exit 1
        ;;
esac