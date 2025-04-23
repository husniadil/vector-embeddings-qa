import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.reasoning import ReasoningTools
from agno.team.team import Team
from agno.knowledge.text import TextKnowledgeBase
from agno.vectordb.lancedb import LanceDb
from agno.embedder.openai import OpenAIEmbedder

# Load environment variables
load_dotenv()


def setup_knowledge_base():
    """Set up a knowledge base using LanceDB and the content from content_data/content.txt"""
    # Create the knowledge base with LanceDB as the vector database
    knowledge_base = TextKnowledgeBase(
        path="content_data",  # Use the content_data directory which contains content.txt
        vector_db=LanceDb(
            table_name="embeddings_team",
            uri="lancedb_data",
            # Use OpenAI embeddings
            embedder=OpenAIEmbedder(
                id=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            ),
        ),
    )

    return knowledge_base


def create_investment_analysis_team(knowledge_base):
    """Create a team of specialized agents for investment analysis"""
    
    # Financial Analyst Agent
    financial_analyst = Agent(
        name="Financial Analyst",
        model=OpenAIChat(id="gpt-3.5-turbo"),
        knowledge=knowledge_base,
        search_knowledge=True,
        tools=[ReasoningTools()],
        instructions=[
            "You are a financial analyst specialized in tech investments.",
            "Your expertise is in analyzing financial metrics, ROI, and investment risks.",
            "Focus on the financial aspects of AI investments, including capital expenditure, expected returns, and financial risks.",
            "Provide quantitative analysis with specific numbers and financial metrics when possible.",
            "Always support your analysis with data from the knowledge base.",
        ],
        markdown=True,
    )
    
    # Technology Analyst Agent
    technology_analyst = Agent(
        name="Technology Analyst",
        model=OpenAIChat(id="gpt-3.5-turbo"),
        knowledge=knowledge_base,
        search_knowledge=True,
        tools=[ReasoningTools()],
        instructions=[
            "You are a technology analyst specialized in AI and infrastructure.",
            "Your expertise is in analyzing technological trends, infrastructure requirements, and technical feasibility.",
            "Focus on the technical aspects of AI investments, including infrastructure needs, technological challenges, and innovation potential.",
            "Evaluate the technical merits and challenges of AI investments.",
            "Always support your analysis with data from the knowledge base.",
        ],
        markdown=True,
    )
    
    # Market Analyst Agent
    market_analyst = Agent(
        name="Market Analyst",
        model=OpenAIChat(id="gpt-3.5-turbo"),
        knowledge=knowledge_base,
        search_knowledge=True,
        tools=[ReasoningTools()],
        instructions=[
            "You are a market analyst specialized in tech industry trends.",
            "Your expertise is in analyzing market dynamics, competitive positioning, and industry trends.",
            "Focus on the market aspects of AI investments, including competitive landscape, market opportunities, and industry trends.",
            "Evaluate how AI investments position companies in the market.",
            "Always support your analysis with data from the knowledge base.",
        ],
        markdown=True,
    )
    
    # Executive Summary Agent
    executive_summary = Agent(
        name="Executive Summary",
        model=OpenAIChat(id="gpt-3.5-turbo"),
        knowledge=knowledge_base,
        search_knowledge=True,
        tools=[ReasoningTools()],
        instructions=[
            "You are an executive communication specialist.",
            "Your expertise is in creating concise, clear summaries for executive audiences.",
            "Focus on distilling complex information into actionable insights.",
            "Create summaries that highlight key points, implications, and recommendations.",
            "Always be concise and use clear, direct language without technical jargon.",
        ],
        markdown=True,
    )
    
    # Create the investment analysis team with coordination mode
    investment_team = Team(
        name="AI Investment Analysis Team",
        mode="coordinate",  # Coordinator delegates tasks and synthesizes responses
        model=OpenAIChat(id="gpt-3.5-turbo"),
        members=[financial_analyst, technology_analyst, market_analyst, executive_summary],
        show_tool_calls=True,
        markdown=True,
        description="You are the coordinator of an AI investment analysis team that provides comprehensive analysis of Big Tech AI investments.",
        instructions=[
            "Coordinate the analysis of AI investments by delegating specific aspects to the appropriate team members:",
            "- Financial Analyst: For financial metrics, ROI, and investment risks",
            "- Technology Analyst: For technological trends, infrastructure requirements, and technical feasibility",
            "- Market Analyst: For market dynamics, competitive positioning, and industry trends",
            "- Executive Summary: For creating concise summaries of the analysis",
            "Based on the user's question, determine which team members should contribute to the analysis.",
            "Synthesize the team members' responses into a comprehensive, well-structured analysis.",
            "For general questions, involve all team members to provide a holistic view.",
            "For specific questions, focus on the most relevant team members.",
            "Always ensure the final response is well-organized with clear sections.",
        ],
        show_members_responses=True,
    )
    
    return investment_team


def main():
    print("Setting up knowledge base...")
    knowledge_base = setup_knowledge_base()

    # Check if we need to load the knowledge base
    # If the vector database already exists, we can skip this step
    if not os.path.exists("lancedb_data/embeddings_team"):
        print("Loading knowledge base...")
        knowledge_base.load(upsert=True)
        print("Knowledge base loaded successfully!")
    else:
        print("Using existing knowledge base.")

    # Create the investment analysis team
    print("Creating AI Investment Analysis Team...")
    investment_team = create_investment_analysis_team(knowledge_base)

    print("\n=== Agno Team Demo ===")
    print("This demo shows how a team of specialized agents can work together to analyze Big Tech AI investments.")
    print("The team consists of four specialized agents:")
    print("1. Financial Analyst: Focuses on financial metrics, ROI, and investment risks")
    print("2. Technology Analyst: Focuses on technological trends, infrastructure, and technical feasibility")
    print("3. Market Analyst: Focuses on market dynamics, competitive positioning, and industry trends")
    print("4. Executive Summary: Creates concise summaries of the analysis")
    print("\nExample questions to ask:")
    print("- 'What are the financial implications of Amazon's $100 billion AI investment?'")
    print("- 'How do Big Tech AI investments impact the competitive landscape?'")
    print("- 'What are the technological challenges of these massive AI infrastructure investments?'")
    print("- 'Provide a comprehensive analysis of Big Tech's AI spending trends.'")

    # Run the team
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == "quit":
            break

        print(f"\nAnalyzing: '{question}'...")
        # Use the team to answer the question
        investment_team.print_response(question, stream=True)


if __name__ == "__main__":
    main()
