#!/bin/bash

# TYPE must be: ["origin", "contextual", "both"]
INGEST_TYPE=$1

FOLDER_DIR=$2

# python script
python source/run/contextual_rag_ingest.py --ingest_type "$INGEST_TYPE" --folder_dir "$FOLDER_DIR" 
