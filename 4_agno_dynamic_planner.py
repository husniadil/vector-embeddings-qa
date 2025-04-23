import os
import json
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool
from agno.tools.reasoning import ReasoningTools
from agno.knowledge.text import TextKnowledgeBase
from agno.vectordb.lancedb import LanceDb
from agno.embedder.openai import OpenAIEmbedder

# Load environment variables
load_dotenv()

completion_model = os.getenv("COMPLETION_MODEL", "gpt-4.1-mini")

# Define a database of Big Tech companies and their AI investments
COMPANY_DATABASE = {
    "microsoft": {
        "name": "Microsoft",
        "planned_investment_2025": "$80 billion",
        "investment_2024": "$53 billion",
        "key_areas": ["Azure Cloud", "OpenAI Partnership", "Copilot Agents"],
        "ceo": "Satya Nadella",
        "stock_impact": "Lost $200 billion in market value after reporting weaker cloud growth",
        "challenges": [
            "Glitchy and costly Copilot agents",
            "Slow enterprise adoption",
            "ROI concerns",
        ],
    },
    "alphabet": {
        "name": "Alphabet (Google)",
        "planned_investment_2025": "$75 billion",
        "investment_2024": "$53 billion",
        "key_areas": ["Gemini AI Models", "Cloud Infrastructure", "AI Research"],
        "ceo": "Sundar Pichai",
        "stock_impact": "8% drop, fifth-worst trading day in past decade",
        "challenges": [
            "Opaque usage metrics for Gemini",
            "Integrating AI into search without cannibalizing ad revenue",
        ],
    },
    "amazon": {
        "name": "Amazon",
        "planned_investment_2025": "$100+ billion",
        "investment_2024": "$77 billion",
        "key_areas": ["AWS Data Centers", "AI Infrastructure", "Specialized Chips"],
        "ceo": "Andy Jassy",
        "stock_impact": "Fell 7% in after-hours trading after investment announcement",
        "challenges": ["Distant ROI", "Significant capital expenditure"],
    },
    "meta": {
        "name": "Meta",
        "planned_investment_2025": "Hundreds of billions",
        "investment_2024": "$40 billion",
        "key_areas": ["AI for Ad Targeting", "Llama Models", "AI Infrastructure"],
        "ceo": "Mark Zuckerberg",
        "stock_impact": "Positive reception, shares rising despite increased spending",
        "challenges": ["Regulatory scrutiny", "Competition in AI space"],
    },
    "openai": {
        "name": "OpenAI",
        "planned_investment_2025": "$100 billion (with partners)",
        "investment_2024": "Not publicly disclosed",
        "key_areas": ["GPT Models", "AI Safety", "US Infrastructure"],
        "ceo": "Sam Altman",
        "stock_impact": "Private company, valued at $260 billion in recent talks",
        "challenges": [
            "Competition from open-source models",
            "Regulatory concerns",
            "Governance issues",
        ],
    },
    "deepseek": {
        "name": "DeepSeek",
        "planned_investment_2025": "Not publicly disclosed",
        "investment_2024": "Not publicly disclosed",
        "key_areas": ["R1 Reasoning Model", "Cost-efficient AI"],
        "ceo": "Not specified",
        "stock_impact": "Caused Nvidia shares to plunge 17%, erasing $600 billion in one day",
        "challenges": [
            "Limited access to advanced Nvidia GPUs",
            "Competition from established players",
        ],
    },
}

# Define a database of AI investment metrics and trends
INVESTMENT_METRICS = {
    "big_tech_combined_2024": "$246 billion",
    "big_tech_combined_2023": "$151 billion",
    "projected_big_tech_2025": "$320+ billion",
    "growth_rate_2023_to_2024": "63%",
    "magnificent_seven_capex_growth": "40%",
    "rest_of_sp500_capex_growth": "3.5%",
    "magnificent_seven_profit_growth": "33%",
    "rest_of_sp500_profit_growth": "5%",
}

# Define a database of AI investment analysis reports
ANALYSIS_REPORTS = {
    "investor_concerns": [
        "Doubling down on spending without commensurate revenue increase",
        "Capital that could be returned as buybacks and dividends",
        "Starving non-AI business lines",
        "Cheaper and more commoditized AI models",
        "Distant return on investment",
    ],
    "executive_responses": [
        "AI opportunity is 'as big as it comes' (Pichai)",
        "Cannot take foot off the gas to remain competitive",
        "Responding to 'significant signals of demand' (Jassy)",
        "Folly of slowing down and failing to capitalize (Nadella)",
        "New techniques could make AI cheaper and spur new research",
    ],
    "analyst_perspectives": [
        "If cloud growth accelerates, investors will be more comfortable with spending",
        "Meta shows tangible returns from AI investment in ad targeting",
        "Google's search empire shows no cracks despite AI competition",
        "Leaders can't take foot off the gas in AI race",
        "Potential for an 'AI winter' at some point, but not imminent",
    ],
}


@tool
def get_company_info(company_name: str) -> str:
    """Get detailed information about a Big Tech company's AI investments.

    Args:
        company_name: The name of the company to look up

    Returns:
        Detailed information about the company's AI investments
    """
    company_name = company_name.lower().strip()

    # Handle common variations
    if company_name == "google":
        company_name = "alphabet"
    elif company_name == "facebook":
        company_name = "meta"

    if company_name in COMPANY_DATABASE:
        company = COMPANY_DATABASE[company_name]
        return json.dumps(company, indent=2)
    else:
        return f"Company '{company_name}' not found in the database. Available companies: {', '.join(COMPANY_DATABASE.keys())}"


@tool
def compare_companies(company1: str, company2: str) -> str:
    """Compare AI investments between two Big Tech companies.

    Args:
        company1: The name of the first company
        company2: The name of the second company

    Returns:
        Comparison of AI investments between the two companies
    """
    company1 = company1.lower().strip()
    company2 = company2.lower().strip()

    # Handle common variations
    if company1 == "google":
        company1 = "alphabet"
    elif company1 == "facebook":
        company1 = "meta"

    if company2 == "google":
        company2 = "alphabet"
    elif company2 == "facebook":
        company2 = "meta"

    if company1 not in COMPANY_DATABASE:
        return f"Company '{company1}' not found in the database. Available companies: {', '.join(COMPANY_DATABASE.keys())}"

    if company2 not in COMPANY_DATABASE:
        return f"Company '{company2}' not found in the database. Available companies: {', '.join(COMPANY_DATABASE.keys())}"

    comp1 = COMPANY_DATABASE[company1]
    comp2 = COMPANY_DATABASE[company2]

    comparison = {
        "comparison": {
            "name": [comp1["name"], comp2["name"]],
            "planned_investment_2025": [
                comp1["planned_investment_2025"],
                comp2["planned_investment_2025"],
            ],
            "investment_2024": [comp1["investment_2024"], comp2["investment_2024"]],
            "key_areas": [comp1["key_areas"], comp2["key_areas"]],
            "ceo": [comp1["ceo"], comp2["ceo"]],
            "stock_impact": [comp1["stock_impact"], comp2["stock_impact"]],
            "challenges": [comp1["challenges"], comp2["challenges"]],
        }
    }

    return json.dumps(comparison, indent=2)


@tool
def get_investment_metrics() -> str:
    """Get overall metrics about Big Tech AI investments.

    Returns:
        Overall metrics and trends in Big Tech AI investments
    """
    return json.dumps(INVESTMENT_METRICS, indent=2)


@tool
def get_investment_analysis(analysis_type: str) -> str:
    """Get analysis about Big Tech AI investments.

    Args:
        analysis_type: Type of analysis to retrieve (investor_concerns, executive_responses, analyst_perspectives)

    Returns:
        Analysis about Big Tech AI investments
    """
    analysis_type = analysis_type.lower().strip()

    if analysis_type in ANALYSIS_REPORTS:
        return json.dumps({analysis_type: ANALYSIS_REPORTS[analysis_type]}, indent=2)
    else:
        return f"Analysis type '{analysis_type}' not found. Available types: {', '.join(ANALYSIS_REPORTS.keys())}"


@tool
def search_content_by_keyword(keyword: str) -> str:
    """Search the content database for information related to a keyword.

    Args:
        keyword: The keyword to search for

    Returns:
        Information related to the keyword from the content database
    """
    # Set up a knowledge base for searching
    knowledge_base = TextKnowledgeBase(
        path="content_data",  # Use the content_data directory which contains content.txt
        vector_db=LanceDb(
            table_name="embeddings_dynamic_planner",
            uri="lancedb_data",  # Use lancedb_data directory for database storage
        ),
        embedder=OpenAIEmbedder(
            id=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        ),
    )

    # Search for the keyword
    results = knowledge_base.search(keyword, limit=3)

    if results:
        return json.dumps({"results": [result.content for result in results]}, indent=2)
    else:
        return f"No information found for keyword '{keyword}'."


@tool
def generate_investment_recommendation(
    company_name: str, investment_amount: float
) -> str:
    """Generate a recommendation for investing in a Big Tech company's AI initiatives.

    Args:
        company_name: The name of the company to analyze
        investment_amount: The amount to potentially invest (in USD)

    Returns:
        Investment recommendation with pros and cons
    """
    company_name = company_name.lower().strip()

    # Handle common variations
    if company_name == "google":
        company_name = "alphabet"
    elif company_name == "facebook":
        company_name = "meta"

    if company_name not in COMPANY_DATABASE:
        return f"Company '{company_name}' not found in the database. Available companies: {', '.join(COMPANY_DATABASE.keys())}"

    company = COMPANY_DATABASE[company_name]

    # Generate a recommendation based on the company's profile
    recommendation = {
        "company": company["name"],
        "investment_amount": f"${investment_amount:,.2f}",
        "recommendation": (
            "Positive" if "Positive" in company["stock_impact"] else "Cautious"
        ),
        "pros": [
            f"Significant AI investment planned for 2025: {company['planned_investment_2025']}",
            f"Focus on key AI areas: {', '.join(company['key_areas'])}",
            f"Strong leadership under CEO {company['ceo']}",
        ],
        "cons": company["challenges"],
        "risk_level": "Medium" if "Positive" in company["stock_impact"] else "High",
        "potential_return": "High but long-term",
        "alternative_considerations": "Consider diversifying AI investments across multiple Big Tech companies",
    }

    return json.dumps(recommendation, indent=2)


@tool
def analyze_market_impact(event: str) -> str:
    """Analyze the market impact of a specific AI-related event or announcement.

    Args:
        event: The AI-related event or announcement to analyze

    Returns:
        Analysis of the market impact
    """
    events = {
        "deepseek r1 release": {
            "description": "Release of DeepSeek's R1 reasoning model that claimed similar capabilities to Google and OpenAI at a fraction of the price",
            "direct_impact": "Caused Nvidia shares to plunge 17%, erasing $600 billion in one day",
            "indirect_impact": "Exacerbated sell-off in Big Tech stocks, particularly Microsoft and Alphabet",
            "long_term_implications": "Raised concerns about AI model commoditization and potential compression of profit margins",
            "expert_opinions": [
                "Could add to demand by showing how new techniques could make AI cheaper (Pichai)",
                "Will probably amplify investor concerns in the meantime (Tierney)",
            ],
        },
        "microsoft earnings": {
            "description": "Microsoft's Q4 2024 earnings report showing weaker than expected cloud growth alongside steep increases in capital spending",
            "direct_impact": "Microsoft had $200 billion wiped from market value",
            "indirect_impact": "Raised investor concerns about the return on AI investments",
            "long_term_implications": "Increased scrutiny on the adoption rate of Microsoft's Copilot agents",
            "expert_opinions": [
                "If we see Copilot uptake improve, investors will be more comfortable with spending (Tierney)"
            ],
        },
        "google earnings": {
            "description": "Google's Q4 2024 earnings report showing 13% growth in ad revenue but opaque metrics about Gemini usage",
            "direct_impact": "Alphabet's 8% drop was its fifth-worst trading day in the past decade",
            "indirect_impact": "Raised questions about Google's ability to monetize its AI investments",
            "long_term_implications": "Increased focus on how Google integrates AI into search without cannibalizing its core ad business",
            "expert_opinions": [
                "If there's meant to be cracks in Google's search empire, it certainly isn't showing up yet (Shmulik)"
            ],
        },
        "amazon investment announcement": {
            "description": "Amazon's announcement of more than $100 billion in capital expenditure for 2025",
            "direct_impact": "Stock fell as much as 7% in after-hours trading",
            "indirect_impact": "Set a new benchmark for AI infrastructure investment among Big Tech",
            "long_term_implications": "Positioned AWS to potentially gain market share in cloud AI services",
            "expert_opinions": [
                "Growth is cooking along a little bit, but the appetite to invest hasn't been curtailed (Pearson)",
                "They are ploughing ahead even if the return on investment seems distant (Pearson)",
            ],
        },
        "meta earnings": {
            "description": "Meta's earnings report and pledge to spend 'hundreds of billions' more on AI",
            "direct_impact": "Shares rising despite increased spending plans",
            "indirect_impact": "Demonstrated that investors can embrace AI spending when ROI is visible",
            "long_term_implications": "Set Meta apart from peers by showing tangible returns from AI investment",
            "expert_opinions": [
                "Investors have embraced Meta because there is a real-time return-on-investment improvement in client spending that is measurable (Tierney)"
            ],
        },
        "openai softbank partnership": {
            "description": "OpenAI's partnership with SoftBank and Oracle to invest $100 billion in AI-related US infrastructure",
            "direct_impact": "Talks to invest $25 billion in OpenAI at a $260 billion valuation",
            "indirect_impact": "Demonstrated continued private investment appetite despite public market concerns",
            "long_term_implications": "Potential to rise to half a trillion investment over time",
            "expert_opinions": [
                "Could there be an AI winter at some point? Sure. But if you're in a position to be a leader, you can't take your foot off the gas (Jaluria)"
            ],
        },
    }

    event = event.lower().strip()

    if event in events:
        return json.dumps(events[event], indent=2)
    else:
        return f"Event '{event}' not found in the database. Available events: {', '.join(events.keys())}"


def setup_knowledge_base():
    """Set up a knowledge base using LanceDB and the content from content_data/content.txt"""
    # Create the knowledge base with LanceDB as the vector database
    knowledge_base = TextKnowledgeBase(
        path="content_data",  # Use the content_data directory which contains content.txt
        vector_db=LanceDb(
            table_name="embeddings_dynamic_planner",
            uri="lancedb_data",  # Use lancedb_data directory for database storage
        ),
        embedder=OpenAIEmbedder(
            id=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        ),
    )

    return knowledge_base


def create_dynamic_planner_agent(knowledge_base):
    """Create a Dynamic Planner Agent for AI investment analysis

    This agent can dynamically plan steps based on the goal.
    It can determine the order of tool calls on its own.
    Example: AI investment analysis with automatic breakdown.
    """
    agent = Agent(
        name="AI Investment Analyst",
        model=OpenAIChat(id=completion_model),
        knowledge=knowledge_base,
        search_knowledge=True,
        tools=[
            ReasoningTools(),
            get_company_info,
            compare_companies,
            get_investment_metrics,
            get_investment_analysis,
            search_content_by_keyword,
            generate_investment_recommendation,
            analyze_market_impact,
        ],
        instructions=[
            "You are an AI investment analyst specializing in Big Tech AI investments.",
            "You can dynamically plan and execute steps based on the user's goal.",
            "First, understand what the user wants to analyze about AI investments.",
            "Then, create a plan with the necessary steps to fulfill their request.",
            "Execute the plan by calling the appropriate tools in the right order.",
            "You can get information about specific companies and their AI investments.",
            "You can compare AI investments between different Big Tech companies.",
            "You can retrieve overall metrics and trends in Big Tech AI investments.",
            "You can get analysis from different perspectives: investors, executives, and analysts.",
            "You can search the content database for specific keywords related to AI investments.",
            "You can generate investment recommendations for specific companies.",
            "You can analyze the market impact of specific AI-related events.",
            "Be proactive and helpful, suggesting additional analysis that might benefit the user.",
            "Always explain your reasoning and plan before taking actions.",
        ],
        markdown=True,
        show_tool_calls=True,
    )

    return agent


def main():
    print("Creating a Dynamic Planner Agent for AI Investment Analysis...")
    knowledge_base = setup_knowledge_base()
    agent = create_dynamic_planner_agent(knowledge_base)

    print("\n=== Dynamic Planner Agent for AI Investment Analysis ===")
    print("This demo shows an agent that can dynamically plan steps based on the goal.")
    print("The agent determines the order of tool calls on its own.")
    print("Suitable for: complex tasks requiring reasoning and multi-step planning.")
    print("\nExample requests:")
    print("- 'Compare Microsoft and Google's AI investments'")
    print("- 'What was the market impact of DeepSeek's R1 release?'")
    print("- 'Should I invest in Meta's AI initiatives?'")
    print("- 'What are the main concerns investors have about Big Tech AI spending?'")
    print("- 'How much are Big Tech companies planning to spend on AI in 2025?'")

    # Run the agent
    while True:
        request = input("\nEnter your request (or 'quit' to exit): ")
        if request.lower() == "quit":
            break

        print(f"\nProcessing request: '{request}'...")
        # Use the agent to process the request
        agent.print_response(request, stream=True)


if __name__ == "__main__":
    main()
