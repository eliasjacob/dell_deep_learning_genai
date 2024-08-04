#!/bin/bash

# Define URLs
URL1="https://public.jacob.al/dell/binaries.zip"

# Define output file names
OUTPUT1="binaries.zip"

# Define output directories
OUTPUT_DIR1="tmp_binaries/"

# Download files
echo "Downloading $URL1..."
curl -o $OUTPUT1 $URL1

# Unzip files into respective directories
echo "Unzipping $OUTPUT1 into $OUTPUT_DIR1 directory..."
mkdir -p $OUTPUT_DIR1
unzip $OUTPUT1 -d $OUTPUT_DIR1

# Move files to the correct directories

# Clean up zip files
rm $OUTPUT1 $OUTPUT2

echo "Download and extraction complete."