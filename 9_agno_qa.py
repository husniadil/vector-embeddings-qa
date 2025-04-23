import json
import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from agno.models.openai import OpenAIChat
from agno.tools.reasoning import ReasoningTools

# Load environment variables
load_dotenv()

# Initialize OpenAI client for embeddings
embedding_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
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


# Load embeddings once at module level for reuse
stored_data = None
try:
    stored_data = load_embeddings()
except Exception as e:
    print(f"Error loading embeddings: {e}")


def get_relevant_context(query: str) -> str:
    """
    Get relevant context for a query using vector search

    Args:
        query (str): The query to search for

    Returns:
        str: The relevant context found in the knowledge base
    """
    global stored_data

    if stored_data is None:
        return "Error: Embeddings not loaded"

    # Create embedding for the query
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    query_embedding_response = embedding_client.embeddings.create(
        model=model, input=query
    )
    query_embedding = query_embedding_response.data[0].embedding

    # Find most relevant context
    relevant_texts = find_most_similar(
        query_embedding, stored_data["embeddings"], stored_data["texts"]
    )

    # Create context from relevant texts
    context = "\n\n".join(relevant_texts)
    return context


def setup_agent():
    """Set up and return an Agno agent for QA"""
    # Define the model to use - use a valid OpenAI model
    model_name = "gpt-4.1-mini"

    # Create the agent with custom tools
    agent = Agent(
        model=OpenAIChat(id=model_name),
        tools=[ReasoningTools(add_instructions=True), get_relevant_context],
        instructions=[
            "You are a helpful assistant. Answer the question based on the provided context.",
            "If the answer cannot be found in the context, say so.",
            "Use the get_relevant_context tool to find information related to the user's question.",
            "Think step by step to provide accurate answers.",
        ],
        markdown=True,
        show_tool_calls=True,
    )

    return agent


def main():
    import os.path

    # Check if embeddings already exist
    if os.path.exists("embedding_data/embedding.json"):
        print("Loading existing embeddings...")
        global stored_data
        stored_data = load_embeddings()
    else:
        # Load content and create embeddings
        texts = load_content()
        print("Creating embeddings...")
        embeddings = create_embeddings(texts)

        # Save embeddings
        print("Saving embeddings...")
        save_embeddings(texts, embeddings)
        stored_data = {"texts": texts, "embeddings": embeddings}

    # Set up the agent
    print("Setting up Agno agent...")
    agent = setup_agent()

    # Example usage of question answering
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == "quit":
            break

        print("\nFinding answer...")
        # Use the agent to answer the question
        agent.print_response(question, stream=True)


if __name__ == "__main__":
    main()
