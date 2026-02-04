#!/bin/bash

# Check if a directory path is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: source script.sh <path_to_parent_folder>"
  # Script finishes here naturally without 'exit'

# Check if the provided path exists and is a directory
elif [ ! -d "$1" ]; then
  echo "Error: '$1' is not a valid directory."
  # Script finishes here naturally without 'exit'

else
  PARENT_DIR="$1"

  # Iterate through each item in the parent directory
  for folder in "$PARENT_DIR"/*; do
    # Check if the item is a directory
    if [ -d "$folder" ]; then
      echo "Running rosbags-convert on: $folder"
      rosbags-convert "$folder"
    fi
  done
fi