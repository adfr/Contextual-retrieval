#!/bin/bash

# Create a new conda environment named 'contextual_retrieval'
if ! conda env list | grep -q "contextual_retrieval"; then
    conda create -n contextual_retrieval python=3.8 -y
fi

# Activate the new environment
source activate contextual_retrieval

# Install requirements from requirements.txt
pip install -r requirements.txt

# Download necessary NLTK data
python -c "import nltk; nltk.download('punkt')"

# Execute the contextual_retrieval.py script
python contextual_retrieval.py



