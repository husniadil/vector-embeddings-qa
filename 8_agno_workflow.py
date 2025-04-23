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


def setup_knowledge_base():
    """Set up a knowledge base using LanceDB and the content from content_data/content.txt"""
    # Create the knowledge base with LanceDB as the vector database
    knowledge_base = TextKnowledgeBase(
        path="content_data",  # Use the content_data directory which contains content.txt
        vector_db=LanceDb(
            table_name="embeddings_workflow",
            uri="sqlite_data",  # Use sqlite_data directory for database storage
            # Use OpenAI embeddings
            embedder=OpenAIEmbedder(
                id=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            ),
        ),
    )

    return knowledge_base


def create_research_agent(knowledge_base):
    """Create an agent for researching information from the knowledge base"""
    return Agent(
        name="Research Agent",
        model=OpenAIChat(id="gpt-3.5-turbo"),
        knowledge=knowledge_base,
        search_knowledge=True,
        tools=[ReasoningTools()],
        instructions=[
            "You are a research assistant specialized in finding information about Big Tech's AI investments.",
            "Your job is to gather relevant facts and data from the knowledge base.",
            "Focus on extracting specific numbers, trends, and company-specific information.",
            "Provide only factual information without analysis or opinions.",
            "Format your response as a structured list of facts.",
        ],
        markdown=True,
    )


def create_analysis_agent():
    """Create an agent for analyzing the research findings"""
    return Agent(
        name="Analysis Agent",
        model=OpenAIChat(id="gpt-3.5-turbo"),
        tools=[ReasoningTools()],
        instructions=[
            "You are a financial analyst specialized in tech investments.",
            "Your job is to analyze the research findings provided to you.",
            "Identify patterns, risks, opportunities, and implications of the AI investments.",
            "Consider market trends, competitive positioning, and potential ROI.",
            "Provide a structured analysis with clear sections.",
        ],
        markdown=True,
    )


def create_summary_agent():
    """Create an agent for creating an executive summary"""
    return Agent(
        name="Summary Agent",
        model=OpenAIChat(id="gpt-3.5-turbo"),
        tools=[ReasoningTools()],
        instructions=[
            "You are an executive communication specialist.",
            "Your job is to create a concise executive summary based on research and analysis.",
            "Focus on the most important points that executives need to know.",
            "Use clear, direct language and avoid technical jargon.",
            "Keep the summary under 250 words.",
            "Include a brief introduction, key findings, and implications.",
        ],
        markdown=True,
    )


def run_workflow(topic, knowledge_base):
    """Run a multi-step workflow to analyze a topic"""
    print(f"\nStep 1: Researching '{topic}'...")
    research_agent = create_research_agent(knowledge_base)
    research_prompt = f"Find information about {topic} in Big Tech AI investments. Focus on specific numbers, trends, and company details."
    research_response = research_agent.run(research_prompt)

    if not research_response or not research_response.content:
        print("Error: Could not complete research step")
        return None

    research_findings = research_response.content
    print("Research step completed")

    print(f"\nStep 2: Analyzing research findings...")
    analysis_agent = create_analysis_agent()
    analysis_prompt = f"Analyze these research findings about {topic} in Big Tech AI investments:\n\n{research_findings}"
    analysis_response = analysis_agent.run(analysis_prompt)

    if not analysis_response or not analysis_response.content:
        print("Error: Could not complete analysis step")
        return None

    analysis = analysis_response.content
    print("Analysis step completed")

    print(f"\nStep 3: Creating executive summary...")
    summary_agent = create_summary_agent()
    summary_prompt = f"Create an executive summary about {topic} in Big Tech AI investments based on this research and analysis:\n\nResearch:\n{research_findings}\n\nAnalysis:\n{analysis}"
    summary_response = summary_agent.run(summary_prompt)

    if not summary_response or not summary_response.content:
        print("Error: Could not complete summary step")
        return None

    summary = summary_response.content
    print("Summary step completed")

    # Save results to files in a dedicated directory
    os.makedirs("sqlite_data/reports", exist_ok=True)
    with open(f"sqlite_data/reports/{topic.replace(' ', '_')}_research.md", "w") as f:
        f.write(research_findings)
    with open(f"sqlite_data/reports/{topic.replace(' ', '_')}_analysis.md", "w") as f:
        f.write(analysis)
    with open(f"sqlite_data/reports/{topic.replace(' ', '_')}_summary.md", "w") as f:
        f.write(summary)

    # Return all results
    return {"research": research_findings, "analysis": analysis, "summary": summary}


def main():
    print("Setting up knowledge base...")
    knowledge_base = setup_knowledge_base()

    # Check if we need to load the knowledge base
    # If the vector database already exists, we can skip this step
    if not os.path.exists("sqlite_data/embeddings_workflow"):
        print("Loading knowledge base...")
        knowledge_base.load(upsert=True)
        print("Knowledge base loaded successfully!")
    else:
        print("Using existing knowledge base.")

    print("\n=== Agno Multi-Agent Workflow Demo ===")
    print(
        "This demo shows how to use multiple Agno agents in a workflow to analyze Big Tech AI investments."
    )
    print("The workflow consists of three steps:")
    print("1. Research: Gather information from the knowledge base")
    print("2. Analysis: Analyze the research findings")
    print("3. Summary: Create an executive summary")
    print("\nExample topics to analyze:")
    print("- spending trends")
    print("- investment risks")
    print("- competitive landscape")
    print("- future projections")

    # Run the workflow
    while True:
        topic = input("\nEnter a topic to analyze (or 'quit' to exit): ")
        if topic.lower() == "quit":
            break

        print(f"\nAnalyzing '{topic}' through the workflow...")
        # Run the workflow with the topic as input
        result = run_workflow(topic, knowledge_base)

        if not result:
            print("Workflow execution failed.")
            continue

        # Print the results from each step
        print("\n=== Research Findings ===")
        print(result["research"])
        print("\n=== Analysis ===")
        print(result["analysis"])
        print("\n=== Executive Summary ===")
        print(result["summary"])


if __name__ == "__main__":
    main()
