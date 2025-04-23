import os
from typing import Dict, Iterator, Optional
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.reasoning import ReasoningTools
from agno.knowledge.text import TextKnowledgeBase
from agno.vectordb.lancedb import LanceDb
from agno.embedder.openai import OpenAIEmbedder
from agno.workflow import Workflow, RunResponse, RunEvent
from agno.storage.sqlite import SqliteStorage
from agno.utils.log import logger

# Load environment variables
load_dotenv()

completion_model = os.getenv("COMPLETION_MODEL", "gpt-4.1-mini")


class AIInvestmentWorkflow(Workflow):
    """Workflow for analyzing Big Tech AI investments using multiple specialized agents"""

    description: str = """
    An intelligent AI investment analysis workflow that creates comprehensive reports on Big Tech AI investments.
    This workflow orchestrates multiple AI agents to research, analyze, and summarize information about
    AI investments, providing valuable insights for decision makers.
    """

    # Research Agent: Handles information gathering from the knowledge base
    research_agent: Agent = None

    # Analysis Agent: Analyzes the research findings
    analysis_agent: Agent = None

    # Summary Agent: Creates executive summaries
    summary_agent: Agent = None

    def __init__(self, session_id: str = None, **kwargs):
        super().__init__(session_id=session_id, **kwargs)

        # Set up knowledge base
        self.knowledge_base = self._setup_knowledge_base()

        # Initialize agents
        self.research_agent = self._create_research_agent()
        self.analysis_agent = self._create_analysis_agent()
        self.summary_agent = self._create_summary_agent()

        # Check if we need to load the knowledge base
        # If the vector database already exists, we can skip this step
        if not os.path.exists("sqlite_data/embeddings_workflow"):
            logger.info("Loading knowledge base...")
            self.knowledge_base.load(upsert=True)
            logger.info("Knowledge base loaded successfully!")
        else:
            logger.info("Using existing knowledge base.")

    def _setup_knowledge_base(self):
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

    def _create_research_agent(self):
        """Create an agent for researching information from the knowledge base"""
        return Agent(
            name="Research Agent",
            model=OpenAIChat(id=completion_model),
            knowledge=self.knowledge_base,
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

    def _create_analysis_agent(self):
        """Create an agent for analyzing the research findings"""
        return Agent(
            name="Analysis Agent",
            model=OpenAIChat(id=completion_model),
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

    def _create_summary_agent(self):
        """Create an agent for creating an executive summary"""
        return Agent(
            name="Summary Agent",
            model=OpenAIChat(id=completion_model),
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

    def get_cached_results(self, topic: str) -> Optional[Dict]:
        """Get cached results for a topic if they exist"""
        logger.info(f"Checking if cached results exist for topic: {topic}")
        return self.session_state.get("results", {}).get(topic)

    def save_results_to_cache(self, topic: str, results: Dict):
        """Save results to cache for future use"""
        logger.info(f"Saving results for topic: {topic}")
        self.session_state.setdefault("results", {})
        self.session_state["results"][topic] = results

    def run(self, topic: str, use_cache: bool = True) -> Iterator[RunResponse]:
        """Run the workflow to analyze a topic"""
        logger.info(f"Starting workflow for topic: {topic}")

        # Check cache first if use_cache is True
        if use_cache:
            cached_results = self.get_cached_results(topic)
            if cached_results:
                logger.info(f"Found cached results for topic: {topic}")
                yield RunResponse(content=cached_results, event=RunEvent.run_completed)
                return

        # Step 1: Research
        logger.info(f"Step 1: Researching '{topic}'...")
        yield RunResponse(
            content=f"Researching '{topic}'...", event=RunEvent.run_started
        )

        research_prompt = f"Find information about {topic} in Big Tech AI investments. Focus on specific numbers, trends, and company details."
        research_response = self.research_agent.run(research_prompt)

        if not research_response or not research_response.content:
            logger.error("Could not complete research step")
            yield RunResponse(
                content="Error: Could not complete research step",
                event=RunEvent.run_error,
            )
            return

        research_findings = research_response.content
        logger.info("Research step completed")
        yield RunResponse(
            content="Research step completed", event=RunEvent.run_response
        )

        # Step 2: Analysis
        logger.info(f"Step 2: Analyzing research findings...")
        yield RunResponse(
            content="Analyzing research findings...", event=RunEvent.run_response
        )

        analysis_prompt = f"Analyze these research findings about {topic} in Big Tech AI investments:\n\n{research_findings}"
        analysis_response = self.analysis_agent.run(analysis_prompt)

        if not analysis_response or not analysis_response.content:
            logger.error("Could not complete analysis step")
            yield RunResponse(
                content="Error: Could not complete analysis step",
                event=RunEvent.run_error,
            )
            return

        analysis = analysis_response.content
        logger.info("Analysis step completed")
        yield RunResponse(
            content="Analysis step completed", event=RunEvent.run_response
        )

        # Step 3: Summary
        logger.info(f"Step 3: Creating executive summary...")
        yield RunResponse(
            content="Creating executive summary...", event=RunEvent.run_response
        )

        summary_prompt = f"Create an executive summary about {topic} in Big Tech AI investments based on this research and analysis:\n\nResearch:\n{research_findings}\n\nAnalysis:\n{analysis}"
        summary_response = self.summary_agent.run(summary_prompt)

        if not summary_response or not summary_response.content:
            logger.error("Could not complete summary step")
            yield RunResponse(
                content="Error: Could not complete summary step",
                event=RunEvent.run_error,
            )
            return

        summary = summary_response.content
        logger.info("Summary step completed")
        yield RunResponse(content="Summary step completed", event=RunEvent.run_response)

        # Save results to files in a dedicated directory
        os.makedirs("sqlite_data/reports", exist_ok=True)
        with open(
            f"sqlite_data/reports/{topic.replace(' ', '_')}_research.md", "w"
        ) as f:
            f.write(research_findings)
        with open(
            f"sqlite_data/reports/{topic.replace(' ', '_')}_analysis.md", "w"
        ) as f:
            f.write(analysis)
        with open(
            f"sqlite_data/reports/{topic.replace(' ', '_')}_summary.md", "w"
        ) as f:
            f.write(summary)

        # Prepare results
        results = {
            "research": research_findings,
            "analysis": analysis,
            "summary": summary,
        }

        # Save to cache
        self.save_results_to_cache(topic, results)

        # Return final results
        yield RunResponse(content=results, event=RunEvent.run_completed)


def main():
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

        # Create a URL-safe topic for session ID
        url_safe_topic = topic.lower().replace(" ", "-")

        # Initialize the workflow with SQLite storage for caching
        workflow = AIInvestmentWorkflow(
            session_id=f"ai-investment-analysis-{url_safe_topic}",
            storage=SqliteStorage(
                table_name="ai_investment_workflows",
                db_file="sqlite_data/agno_workflows.db",
            ),
            debug_mode=True,
        )

        print(f"\nAnalyzing '{topic}' through the workflow...")

        # Run the workflow and collect results
        results = None
        for response in workflow.run(topic=topic, use_cache=True):
            if response.event == RunEvent.run_completed:
                results = response.content
            elif response.event == RunEvent.run_error:
                print(f"Workflow failed: {response.content}")
                break
            else:
                print(response.content)

        if results:
            # Print the results from each step
            print("\n=== Research Findings ===")
            print(results["research"])
            print("\n=== Analysis ===")
            print(results["analysis"])
            print("\n=== Executive Summary ===")
            print(results["summary"])


if __name__ == "__main__":
    main()
