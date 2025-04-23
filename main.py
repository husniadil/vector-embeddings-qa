import json
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

completion_model = os.getenv("COMPLETION_MODEL", "gpt-4.1-mini")

# Initialize OpenAI-compatible clients
embedding_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Use OpenRouter if configured, otherwise fall back to OpenAI
completion_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY", os.getenv("OPENAI_API_KEY")),
    base_url=os.getenv(
        "OPENROUTER_BASE_URL",
        os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    ),
)


def load_content():
    """Load content from content_data/content.txt, where each line is a separate text entry"""
    with open("content_data/content.txt", "r") as f:
        return [line.strip() for line in f if line.strip()]


def create_embeddings(texts):
    """Create embeddings for given texts using the configured embedding model"""
    embeddings = []
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    for text in texts:
        response = embedding_client.embeddings.create(model=model, input=text)
        embeddings.append(response.data[0].embedding)
    return embeddings


def save_embeddings(texts, embeddings):
    """Save texts and their embeddings to embedding_data/embedding.json"""
    data = {"texts": texts, "embeddings": embeddings}
    with open("embedding_data/embedding.json", "w") as f:
        json.dump(data, f)


def load_embeddings():
    """Load embeddings from embedding_data/embedding.json"""
    with open("embedding_data/embedding.json", "r") as f:
        return json.load(f)


def find_most_similar(query_embedding, embeddings, texts, top_k=3):
    """Find most similar texts based on cosine similarity"""
    # Convert embeddings to numpy array for efficient computation
    embeddings_array = np.array(embeddings)
    query_array = np.array(query_embedding)

    # Calculate cosine similarity
    similarities = np.dot(embeddings_array, query_array) / (
        np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_array)
    )

    # Get indices of top k similar texts
    top_indices = np.argsort(similarities)[-top_k:]

    return [texts[i] for i in top_indices]


def answer_question(question):
    """Answer a question using embeddings and the configured completion model"""
    # Create embedding for the question
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    question_embedding_response = embedding_client.embeddings.create(
        model=model, input=question
    )
    question_embedding = question_embedding_response.data[0].embedding

    # Load stored embeddings
    stored_data = load_embeddings()

    # Find most relevant context
    relevant_texts = find_most_similar(
        question_embedding, stored_data["embeddings"], stored_data["texts"]
    )

    # Create context from relevant texts
    context = "\n\n".join(relevant_texts)

    # Generate answer using the configured model
    response = completion_client.chat.completions.create(
        model=completion_model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer the question based on the provided context. If the answer cannot be found in the context, say so.",
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )

    return response.choices[0].message.content


def main():
    import os.path

    # Check if embeddings already exist
    if os.path.exists("embedding_data/embedding.json"):
        print("Loading existing embeddings...")
        stored_data = load_embeddings()
        texts = stored_data["texts"]
        embeddings = stored_data["embeddings"]
    else:
        # Load content and create embeddings
        texts = load_content()
        print("Creating embeddings...")
        embeddings = create_embeddings(texts)

        # Save embeddings
        print("Saving embeddings...")
        save_embeddings(texts, embeddings)

    # Example usage of question answering
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == "quit":
            break

        print("\nFinding answer...")
        answer = answer_question(question)
        print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    main()
