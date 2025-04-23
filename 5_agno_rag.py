import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.reasoning import ReasoningTools
from agno.knowledge.text import TextKnowledgeBase
from agno.vectordb.lancedb import LanceDb
from agno.embedder.openai import OpenAIEmbedder

# Load environment variables
load_dotenv()


def load_content():
    """Load content from content_data/content.txt, where each line is a separate text entry"""
    with open("content_data/content.txt", "r") as f:
        return [line.strip() for line in f if line.strip()]


def setup_knowledge_base():
    """Set up a knowledge base using LanceDB and the content from content_data/content.txt"""
    # Create the knowledge base with LanceDB as the vector database
    knowledge_base = TextKnowledgeBase(
        path="content_data",  # Use the content_data directory which contains only content.txt
        vector_db=LanceDb(
            table_name="embeddings",
            uri="lancedb_data",
            # Use OpenAI embeddings
            embedder=OpenAIEmbedder(
                id=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            ),
        ),
    )

    return knowledge_base


def ask_question(question):
    """Ask a question using the Agno agent with RAG"""
    # Set up the knowledge base
    knowledge_base = setup_knowledge_base()

    # Set up the agent
    agent = Agent(
        model=OpenAIChat(id="gpt-3.5-turbo"),
        knowledge=knowledge_base,
        # Enable agentic RAG - let the agent decide when to search the knowledge base
        search_knowledge=True,
        tools=[ReasoningTools(add_instructions=True)],
        instructions=[
            "You are a helpful assistant. Answer questions based on the provided knowledge base.",
            "If the answer cannot be found in the knowledge base, say so.",
            "Think step by step to provide accurate answers.",
            "Use the search_knowledge tool to find information related to the user's question.",
            "The knowledge base contains information about Big Tech's AI investments and spending.",
        ],
        markdown=True,
        show_tool_calls=True,
    )

    # Use the agent to answer the question
    agent.print_response(question, stream=True)


def main():
    print("Setting up knowledge base...")
    knowledge_base = setup_knowledge_base()

    # Check if we need to load the knowledge base
    # If the vector database already exists, we can skip this step
    if not os.path.exists("lancedb_data/embeddings"):
        print("Loading knowledge base...")
        knowledge_base.load(upsert=True)
        print("Knowledge base loaded successfully!")
    else:
        print("Using existing knowledge base.")

    # Example usage of question answering
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == "quit":
            break

        print("\nFinding answer...")
        # Use the agent to answer the question
        if question.lower() in ["summarize", "summary"]:
            # For summarize questions, provide more context
            ask_question(
                "Summarize the information about Big Tech's AI investments and spending"
            )
        else:
            ask_question(question)


if __name__ == "__main__":
    main()
