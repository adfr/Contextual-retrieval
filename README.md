# Contextual Retrieval System

This project implements a Retrieval-Augmented Generation (RAG) system using Wikipedia content, ChromaDB for vector storage, and OpenAI's GPT model for question answering.

## Prerequisites

- Python 3.8 or higher
- Conda (for environment management)
- OpenAI API key

## Setup

1. Clone this repository to your local machine.

2. Create a file named `api_key.txt` in the project root directory and paste your OpenAI API key into it.

3. Run the setup script:

   ```
   bash run.sh
   ```

   This script will:
   - Create a new conda environment named 'contextual_retrieval'
   - Install all required dependencies
   - Download necessary NLTK data
   - Set up the OPENAI_API_KEY environment variable

## Usage

After running the setup script, the main program `contextual_retrieval.py` will execute automatically. This script will:

1. Fetch content from Wikipedia for a predefined list of AI-related topics.
2. Process and chunk the content.
3. Generate summaries for each topic (if enabled).
4. Store the processed content in a ChromaDB collection.
5. Build a RAG system using the stored content.
6. Run example queries to demonstrate the system's functionality.

## Customization

- To add or remove topics, modify the `topics` list in `contextual_retrieval.py`.
- To adjust the chunking process, modify the `chunk_size` and `overlap` parameters in the `chunk_text` function.
- To enable or disable summary generation, change the `summary_bool` parameter when calling `process_wikipedia_page`.

## Note

Ensure that your OpenAI API key has sufficient credits and permissions to use the required models (GPT-4 and text-embedding-ada-002).

## Troubleshooting

If you encounter any issues related to the OpenAI API key, make sure:
1. The `api_key.txt` file exists and contains a valid API key.
2. The `run.sh` script has execute permissions (`chmod +x run.sh`).
3. The OPENAI_API_KEY environment variable is correctly set.

For any other issues, please check the error messages and ensure all dependencies are correctly installed.
