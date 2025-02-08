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

## Sample Output
###### (with OpenRouter using google/gemini-2.0-flash-001 model)
```bash
(.venv) (.venv)vector-embeddings-qa ➤ python main.py
Loading existing embeddings...

Enter your question (or 'quit' to exit): who are you? who developed you?                                   

Finding answer...

Answer: I am a helpful assistant. I was developed by Google.


Enter your question (or 'quit' to exit): How much has the alphabet dropped?

Finding answer...

Answer: Alphabet dropped 8 percent.


Enter your question (or 'quit' to exit): summarize

Finding answer...

Answer: Here is a summary of the provided text:

Jim Tierney of AllianceBernstein notes that the enthusiasm for the "Magnificent Seven" stocks is waning. This group, which includes Apple, Nvidia, and Tesla, significantly outspends the rest of the S&P 500, with a 40% increase in capital spending in 2024 compared to just 3.5% for the other 493 companies. Their profits also grew substantially more, at 33% versus 5% for the rest. Google is also mentioned for introducing "AI overviews" in its search results, which are displacing traditional search result links.


Enter your question (or 'quit' to exit): quit
```

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