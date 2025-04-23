import json
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool

# Load environment variables
load_dotenv()


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


def create_tool_calling_agent():
    """Create a Tool-Calling Agent (Single-Step)

    This agent can call a single tool in one step.
    No planning or context memory.
    Suitable for: simple information retrieval, database queries, etc.
    """
    agent = Agent(
        name="AI Investment Information Agent",
        model=OpenAIChat(id="gpt-3.5-turbo"),
        # No knowledge base
        # No memory
        tools=[get_company_info, compare_companies, get_investment_metrics],
        instructions=[
            "You are an AI investment information agent specializing in Big Tech AI investments.",
            "You can answer questions about AI investments by Big Tech companies.",
            "You can provide information about specific companies like Microsoft, Google, Amazon, Meta, etc.",
            "You can compare AI investments between different companies.",
            "You can provide overall metrics about Big Tech AI investments.",
            "Use the appropriate tool to answer the user's question.",
            "For company-specific information, use the get_company_info tool.",
            "For comparing two companies, use the compare_companies tool.",
            "For overall investment metrics, use the get_investment_metrics tool.",
            "Always explain the information you provide in a clear and concise manner.",
            "If you don't have a tool to answer a specific question, just say so.",
        ],
        markdown=True,
        show_tool_calls=True,
    )

    return agent


def main():
    print("Creating a Tool-Calling Agent (Single-Step)...")
    agent = create_tool_calling_agent()

    print("\n=== Tool-Calling Agent Demo ===")
    print("This demo shows an agent that can call a single tool in one step.")
    print("No planning or context memory.")
    print("Suitable for: simple information retrieval, database queries, etc.")
    print("\nExample requests:")
    print("- 'Tell me about Microsoft's AI investments'")
    print("- 'Compare Google and Meta's AI investments'")
    print("- 'What are the overall metrics for Big Tech AI investments?'")
    print("- 'How much is Amazon planning to invest in AI in 2025?'")
    print("- 'What are the challenges Microsoft faces with its AI investments?'")

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
