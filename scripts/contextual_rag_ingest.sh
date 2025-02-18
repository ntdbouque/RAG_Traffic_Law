#!/bin/bash

# TYPE must be: ["origin", "contextual", "both"]
TYPE=$1

FOLDER_DIR=$2

# python script
python source/run/contextual_rag_ingest.py --type "$TYPE" --folder_dir "$FOLDER_DIR" 
