# Vector-Based Question Answering System

This project implements a vector-based question answering system using OpenAI's embeddings and configurable language models. It allows users to ask questions about a given text content and receive relevant answers based on semantic similarity. The system supports both OpenAI and OpenRouter as LLM providers.

## Overview

The system works by:
1. Converting text content into vector embeddings using OpenAI's text-embedding-3-small model
2. Storing these embeddings for efficient retrieval
3. Processing user questions by converting them to embeddings and finding the most relevant context
4. Generating answers using a configurable language model (default: gpt-4o-mini) based on the retrieved context

## Prerequisites

- Python 3.x
- OpenAI API key (required for embeddings)
- OpenRouter API key (optional, for alternative language models)

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the example environment file and configure it with your settings:
   ```bash
   cp .env.example .env
   ```
   Then edit the `.env` file and update the configuration values according to your needs. The example file contains all the necessary configuration options with detailed comments.

## Usage

1. Prepare your content in a `content.txt` file, with each line representing a separate text segment
2. Run the main script:
   ```bash
   python main.py
   ```
3. The script will:
   - Load the content
   - Create embeddings
   - Save them for future use
   - Start an interactive question-answering session

## Technical Implementation

### Key Components

- **Embedding Generation**: Uses OpenAI's text-embedding-3-small model to convert text into vector representations
- **Similarity Search**: Implements cosine similarity to find the most relevant context for each question
- **Answer Generation**: Utilizes configurable language models (default: gpt-4o-mini) to generate accurate answers based on the retrieved context

### File Structure

- `main.py`: Core implementation of the question-answering system
- `requirements.txt`: Project dependencies
- `content.txt`: Source content for embeddings (one text segment per line)
- `embedding.json`: Generated embeddings storage

## Attribution

The content used in this project is sourced from the Financial Times article:
["Big Tech lines up over $300bn in AI spending for 2025"](https://www.ft.com/content/634b7ec5-10c3-44d3-ae49-2a5b9ad566fa)

## License

This project is for educational purposes only. Please ensure you comply with OpenAI's usage terms, OpenRouter's terms of service (if using their models), and the Financial Times' content usage policies.