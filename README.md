# Traffic Law Retrieval using Contextual RAG

This project focuses on retrieving traffic law information using Contextual Retrieval-Augmented Generation (RAG). The system leverages advanced machine learning models to provide accurate and contextually relevant legal information.

## Installation

### Requirements

- Python 3.9 or higher
- Docker
- OpenAI API Key
- LlamaParser API Key

### Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/traffic-law-retrieval.git
    cd traffic-law-retrieval
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your OpenAI API key:
    ```bash
    export OPENAI_API_KEY='your-openai-api-key'
    ```

4. Set up your LlamaParser API key:
    ```bash
    export LLAMA_PARSER_API_KEY='your-llama-parser-api-key'
    ```

5. Run Docker Compose to start Qdrant and Elasticsearch:
    ```bash
    docker-compose up
    ```

## Usage

### Ingest Data

To ingest data into the system, run the following command:
```bash
python ingest.py
```

This will process and store the necessary data for retrieval.

### Query the System

You can query the system using the provided interface or API endpoints to retrieve relevant traffic law information based on your input.
