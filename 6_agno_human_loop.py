import os
import json
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.reasoning import ReasoningTools
from agno.knowledge.text import TextKnowledgeBase
from agno.vectordb.lancedb import LanceDb
from agno.embedder.openai import OpenAIEmbedder
from agno.tools import FunctionCall, tool
from agno.exceptions import StopAgentRun
from rich.console import Console
from rich.prompt import Prompt

# Load environment variables
load_dotenv()

# Create a console for rich output
console = Console()

completion_model = os.getenv("COMPLETION_MODEL", "gpt-4.1-mini")


def load_content():
    """Load content from content_data/content.txt, where each line is a separate text entry"""
    with open("content_data/content.txt", "r") as f:
        return [line.strip() for line in f if line.strip()]


def setup_knowledge_base():
    """Set up a knowledge base using LanceDB and the content from content_data/content.txt"""
    # Create the knowledge base with LanceDB as the vector database
    knowledge_base = TextKnowledgeBase(
        path="content_data",  # Use the content_data directory which contains content.txt
        vector_db=LanceDb(
            table_name="embeddings_human_loop",
            uri="lancedb_data",
            # Use OpenAI embeddings
            embedder=OpenAIEmbedder(
                id=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            ),
        ),
    )

    return knowledge_base


def human_confirmation_hook(fc: FunctionCall):
    """Pre-hook for tool calls to get human confirmation"""
    # Get the live display instance from the console
    live = console._live

    # Stop the live display temporarily so we can ask for user confirmation
    if live:
        live.stop()  # type: ignore

    # Ask for confirmation
    console.print(f"\n[bold blue]AI wants to perform: {fc.function.name}[/]")
    console.print(f"[bold green]Parameters:[/] {json.dumps(fc.arguments, indent=2)}")
    message = (
        Prompt.ask("Do you want to allow this action?", choices=["y", "n"], default="y")
        .strip()
        .lower()
    )

    # Restart the live display
    if live:
        live.start()  # type: ignore

    # If the user does not want to continue, raise a StopAgentRun exception
    if message != "y":
        raise StopAgentRun(
            "Action cancelled by user",
            agent_message="The user has declined this action. Let's try a different approach or ask for more information.",
        )


@tool(pre_hook=human_confirmation_hook)
def get_human_input(question: str) -> str:
    """Ask the human for input or clarification.

    Args:
        question (str): The question to ask the human

    Returns:
        str: The human's response
    """
    # Get the live display instance from the console
    live = console._live

    # Stop the live display temporarily
    if live:
        live.stop()  # type: ignore

    # Ask the human for input
    console.print(f"\n[bold green]AI is asking:[/] {question}")
    response = Prompt.ask("Your response")

    # Restart the live display
    if live:
        live.start()  # type: ignore

    return response


@tool(pre_hook=human_confirmation_hook)
def analyze_investment_data(company: str, investment_amount: str) -> str:
    """Analyze investment data for a specific company.

    Args:
        company (str): The name of the company to analyze
        investment_amount (str): The investment amount to analyze

    Returns:
        str: Analysis of the investment data
    """
    # This is a mock function that would normally do some real analysis
    # For demo purposes, we'll just return some mock data
    analysis = {
        "company": company,
        "investment_amount": investment_amount,
        "risk_level": "medium to high",
        "potential_roi": "15-20% over 3 years",
        "market_impact": "Significant potential to reshape the AI infrastructure landscape",
        "competitive_analysis": "Leading position against similar investments from competitors",
    }

    return json.dumps(analysis, indent=2)


def setup_agent_with_human_loop(knowledge_base):
    """Set up an Agno agent with human-in-the-loop capability"""
    # Create the agent with the knowledge base and human input tools
    agent = Agent(
        model=OpenAIChat(id=completion_model),
        knowledge=knowledge_base,
        # Enable agentic RAG - let the agent decide when to search the knowledge base
        search_knowledge=True,
        tools=[
            ReasoningTools(add_instructions=True),
            get_human_input,
            analyze_investment_data,
        ],
        instructions=[
            "You are a helpful assistant that can answer questions about Big Tech's AI investments and spending.",
            "If the answer cannot be found in the knowledge base, you can ask the human for clarification or additional information using the get_human_input tool.",
            "Use the search_knowledge tool to find information in the knowledge base.",
            "Use the analyze_investment_data tool when you need to provide detailed analysis of a company's AI investments.",
            "Always think step by step to provide accurate answers.",
            "After receiving human input, incorporate it into your reasoning and response.",
        ],
        markdown=True,
        show_tool_calls=True,
    )

    return agent


def main():
    print("Setting up knowledge base...")
    knowledge_base = setup_knowledge_base()

    # Check if we need to load the knowledge base
    # If the vector database already exists, we can skip this step
    if not os.path.exists("lancedb_data/embeddings_human_loop"):
        print("Loading knowledge base...")
        knowledge_base.load(upsert=True)
        print("Knowledge base loaded successfully!")
    else:
        print("Using existing knowledge base.")

    # Set up the agent with human-in-the-loop capability
    print("Setting up Agno agent with human-in-the-loop capability...")
    agent = setup_agent_with_human_loop(knowledge_base)

    print("\n=== Human-in-the-Loop Demo ===")
    print(
        "This demo shows how an AI agent can interact with both a knowledge base and a human."
    )
    print(
        "The agent will search the knowledge base for information, but can also ask you for"
    )
    print("clarification or additional details when needed.")
    print("Try asking complex questions that might require clarification or follow-up.")
    print("Example: 'What are the potential risks of these AI investments?'")
    print("Example: 'How do these investments compare to historical tech spending?'")
    print("Example: 'What's the expected ROI timeline for these companies?'")
    print("Example: 'Analyze Amazon's $100 billion investment'")

    # Example usage of question answering with human in the loop
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == "quit":
            break

        print("\nFinding answer...")
        # Use the agent to answer the question with potential human interaction
        agent.print_response(question, stream=True, console=console)


if __name__ == "__main__":
    main()
