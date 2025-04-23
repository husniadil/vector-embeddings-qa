import json
import time
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
}

# Define a database of investment reports
REPORT_DATABASE = {}


@tool
def search_companies_by_investment_size(min_amount: str) -> str:
    """Search for companies with investment plans above a certain amount.

    Args:
        min_amount: The minimum investment amount (e.g., "70 billion")

    Returns:
        List of companies with investments above the specified amount
    """
    # Extract the numeric part from the input
    try:
        # Remove non-numeric characters except decimal point
        amount_str = "".join(c for c in min_amount if c.isdigit() or c == ".")
        min_investment = float(amount_str)
    except ValueError:
        return f"Invalid amount format: {min_amount}. Please provide a number followed by 'billion' (e.g., '70 billion')."

    matching_companies = {}
    for company_id, company in COMPANY_DATABASE.items():
        # Extract numeric part from planned investment
        planned_investment = company["planned_investment_2025"]
        if "billion" in planned_investment.lower():
            try:
                # Extract numeric value
                investment_value = float(
                    "".join(c for c in planned_investment if c.isdigit() or c == ".")
                )
                if investment_value >= min_investment:
                    matching_companies[company_id] = company
            except ValueError:
                # Skip if we can't parse the investment value
                continue

    if matching_companies:
        return json.dumps(matching_companies, indent=2)
    else:
        return f"No companies found with planned investments above {min_amount}."


@tool
def get_company_details(company_id: str) -> str:
    """Get detailed information about a company.

    Args:
        company_id: The ID of the company to look up

    Returns:
        Detailed information about the company
    """
    company_id = company_id.lower().strip()

    # Handle common variations
    if company_id == "google":
        company_id = "alphabet"
    elif company_id == "facebook":
        company_id = "meta"

    if company_id in COMPANY_DATABASE:
        company = COMPANY_DATABASE[company_id]
        return json.dumps(company, indent=2)
    else:
        return f"Company with ID '{company_id}' not found in the database."


@tool
def analyze_investment_risks(company_id: str) -> str:
    """Analyze the investment risks for a company.

    Args:
        company_id: The ID of the company to analyze

    Returns:
        Risk analysis for the company's AI investments
    """
    company_id = company_id.lower().strip()

    # Handle common variations
    if company_id == "google":
        company_id = "alphabet"
    elif company_id == "facebook":
        company_id = "meta"

    if company_id not in COMPANY_DATABASE:
        return f"Company with ID '{company_id}' not found in the database."

    company = COMPANY_DATABASE[company_id]

    # Generate a risk analysis based on the company's profile
    risk_analysis = {
        "company": company["name"],
        "investment_size": company["planned_investment_2025"],
        "risk_factors": company["challenges"],
        "market_reaction": company["stock_impact"],
        "risk_level": (
            "High"
            if "Lost" in company["stock_impact"]
            or "drop" in company["stock_impact"]
            or "Fell" in company["stock_impact"]
            else "Medium"
        ),
        "potential_mitigations": [
            "Clearer communication about ROI expectations",
            "Phased investment approach with measurable milestones",
            "Strategic partnerships to share investment burden",
            "Focus on areas with demonstrated market traction",
        ],
    }

    return json.dumps(risk_analysis, indent=2)


@tool
def analyze_investment_opportunities(company_id: str) -> str:
    """Analyze the investment opportunities for a company.

    Args:
        company_id: The ID of the company to analyze

    Returns:
        Opportunity analysis for the company's AI investments
    """
    company_id = company_id.lower().strip()

    # Handle common variations
    if company_id == "google":
        company_id = "alphabet"
    elif company_id == "facebook":
        company_id = "meta"

    if company_id not in COMPANY_DATABASE:
        return f"Company with ID '{company_id}' not found in the database."

    company = COMPANY_DATABASE[company_id]

    # Generate an opportunity analysis based on the company's profile
    opportunity_analysis = {
        "company": company["name"],
        "investment_size": company["planned_investment_2025"],
        "key_opportunity_areas": company["key_areas"],
        "competitive_advantages": [
            f"Strong leadership under {company['ceo']}",
            f"Significant financial commitment: {company['planned_investment_2025']}",
            "Established market position",
            "Technical expertise and talent pool",
        ],
        "potential_growth_vectors": [
            "Enterprise AI adoption acceleration",
            "New AI-powered product categories",
            "Efficiency gains in existing operations",
            "Market share expansion in cloud and AI services",
        ],
        "opportunity_level": (
            "High" if "Positive" in company["stock_impact"] else "Medium"
        ),
    }

    return json.dumps(opportunity_analysis, indent=2)


@tool
def generate_investment_report(
    company_id: str, analyst_name: str, recommendation: str
) -> str:
    """Generate a comprehensive investment report for a company.

    Args:
        company_id: The ID of the company to report on
        analyst_name: The name of the analyst creating the report
        recommendation: The investment recommendation (Buy, Hold, Sell)

    Returns:
        Confirmation of report generation
    """
    company_id = company_id.lower().strip()

    # Handle common variations
    if company_id == "google":
        company_id = "alphabet"
    elif company_id == "facebook":
        company_id = "meta"

    if company_id not in COMPANY_DATABASE:
        return f"Company with ID '{company_id}' not found in the database."

    company = COMPANY_DATABASE[company_id]

    # Generate a report ID
    report_id = f"REP-{int(time.time())}"

    # Create the report
    REPORT_DATABASE[report_id] = {
        "report_id": report_id,
        "company_id": company_id,
        "company_name": company["name"],
        "analyst_name": analyst_name,
        "recommendation": recommendation,
        "investment_size": company["planned_investment_2025"],
        "key_areas": company["key_areas"],
        "challenges": company["challenges"],
        "market_reaction": company["stock_impact"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "completed",
    }

    return f"Investment report generated successfully! Report ID: {report_id}. The report for {company['name']}'s AI investments has been created by analyst {analyst_name} with a {recommendation} recommendation."


def run_hardcoded_workflow(min_investment, agent):
    """Run a hardcoded workflow with predefined steps:
    1. Search for companies with investments above a threshold
    2. Get details for a specific company
    3. Analyze investment risks
    4. Analyze investment opportunities
    5. Generate investment report
    """
    print(
        f"\nStep 1: Searching for companies with investments above {min_investment}..."
    )
    search_prompt = f"Search for companies planning to invest more than {min_investment} in AI in 2025"
    search_result = agent.run(search_prompt)
    
    # Display the search results
    if search_result and search_result.content:
        print("\nSearch Results:")
        print(search_result.content)
        
        # Try to parse the JSON response
        try:
            companies_data = json.loads(search_result.content)
            if not companies_data:
                print(f"No companies found with planned investments above {min_investment}.")
                return
                
            # Get the first company ID from the search results
            company_id = list(companies_data.keys())[0]
            company_name = companies_data[company_id]["name"]
            
            print(f"Found company: {company_name} (ID: {company_id})")
            
            print(f"\nStep 2: Getting details for '{company_name}'...")
            details_prompt = f"Get details for company {company_id}"
            details_result = agent.run(details_prompt)
            
            # Display the company details
            if details_result and details_result.content:
                print("\nCompany Details:")
                print(details_result.content)
                
                print(f"\nStep 3: Analyzing investment risks for '{company_name}'...")
                risks_prompt = f"Analyze investment risks for {company_id}"
                risks_result = agent.run(risks_prompt)
                
                # Display the risk analysis
                if risks_result and risks_result.content:
                    print("\nRisk Analysis:")
                    print(risks_result.content)
                    
                    print(f"\nStep 4: Analyzing investment opportunities for '{company_name}'...")
                    opportunities_prompt = f"Analyze investment opportunities for {company_id}"
                    opportunities_result = agent.run(opportunities_prompt)
                    
                    # Display the opportunity analysis
                    if opportunities_result and opportunities_result.content:
                        print("\nOpportunity Analysis:")
                        print(opportunities_result.content)
                        
                        print(f"\nStep 5: Generating investment report for '{company_name}'...")
                        report_prompt = f"Generate investment report for {company_id} by analyst John Smith with a Hold recommendation"
                        report_result = agent.run(report_prompt)
                        
                        # Display the investment report
                        if report_result and report_result.content:
                            print("\nInvestment Report:")
                            print(report_result.content)
                            
                            print("\nWorkflow completed successfully!")
                        else:
                            print("Error: Could not generate investment report.")
                    else:
                        print("Error: Could not analyze investment opportunities.")
                else:
                    print("Error: Could not analyze investment risks.")
            else:
                print("Error: Could not get company details.")
        except json.JSONDecodeError:
            # Fall back to the original approach if we can't parse the JSON
            fallback_to_database_approach(min_investment, agent)
    else:
        print("Error: Could not search for companies.")
        # Fall back to the original approach
        fallback_to_database_approach(min_investment, agent)


def fallback_to_database_approach(min_investment, agent):
    """Fallback method that uses the database directly if the agent response can't be parsed"""
    # Extract the first company ID directly from the database
    matching_companies = {}
    for company_id, company in COMPANY_DATABASE.items():
        # Extract numeric part from planned investment
        planned_investment = company["planned_investment_2025"]
        if "billion" in planned_investment.lower():
            try:
                # Extract numeric value
                investment_value = float(
                    "".join(c for c in planned_investment if c.isdigit() or c == ".")
                )
                if investment_value >= float(min_investment.split()[0]):
                    matching_companies[company_id] = company
            except ValueError:
                # Skip if we can't parse the investment value
                continue

    if not matching_companies:
        print(f"No companies found with planned investments above {min_investment}.")
        return

    # Get the first company ID from the matching companies
    company_id = list(matching_companies.keys())[0]
    company_name = matching_companies[company_id]["name"]

    print(f"Found company: {company_name} (ID: {company_id})")

    print(f"\nStep 2: Getting details for '{company_name}'...")
    details_prompt = f"Get details for company {company_id}"
    details_result = agent.run(details_prompt)
    
    # Display the company details
    if details_result and details_result.content:
        print("\nCompany Details:")
        print(details_result.content)

    print(f"\nStep 3: Analyzing investment risks for '{company_name}'...")
    risks_prompt = f"Analyze investment risks for {company_id}"
    risks_result = agent.run(risks_prompt)
    
    # Display the risk analysis
    if risks_result and risks_result.content:
        print("\nRisk Analysis:")
        print(risks_result.content)

    print(f"\nStep 4: Analyzing investment opportunities for '{company_name}'...")
    opportunities_prompt = f"Analyze investment opportunities for {company_id}"
    opportunities_result = agent.run(opportunities_prompt)
    
    # Display the opportunity analysis
    if opportunities_result and opportunities_result.content:
        print("\nOpportunity Analysis:")
        print(opportunities_result.content)

    print(f"\nStep 5: Generating investment report for '{company_name}'...")
    report_prompt = f"Generate investment report for {company_id} by analyst John Smith with a Hold recommendation"
    report_result = agent.run(report_prompt)
    
    # Display the investment report
    if report_result and report_result.content:
        print("\nInvestment Report:")
        print(report_result.content)

    print("\nWorkflow completed successfully!")


def create_hardcoded_multi_tool_agent():
    """Create a Hardcoded Multi-Tool Agent

    This agent follows a predefined sequence of steps (static flow).
    Suitable for simple automation processes (RPA style).
    Example: "Search companies → check details → analyze risks → analyze opportunities → generate report."
    """
    agent = Agent(
        name="AI Investment Analyst",
        model=OpenAIChat(id="gpt-4.1-mini"),
        # No knowledge base
        # No memory
        tools=[
            search_companies_by_investment_size,
            get_company_details,
            analyze_investment_risks,
            analyze_investment_opportunities,
            generate_investment_report,
        ],
        instructions=[
            "You are an AI investment analyst specializing in Big Tech AI investments.",
            "You follow a specific sequence of steps to analyze AI investments.",
            "You can search for companies based on their investment size.",
            "You can get detailed information about specific companies.",
            "You can analyze investment risks for companies.",
            "You can analyze investment opportunities for companies.",
            "You can generate comprehensive investment reports.",
            "Follow the user's instructions for each step of the analysis process.",
        ],
        markdown=True,
        show_tool_calls=True,
    )

    return agent


def main():
    print("Creating a Hardcoded Multi-Tool Agent...")
    agent = create_hardcoded_multi_tool_agent()

    print("\n=== Hardcoded Multi-Tool Agent Demo ===")
    print(
        "This demo shows an agent that follows a predefined sequence of steps (static flow)."
    )
    print("Suitable for simple automation processes (RPA style).")
    print(
        "The workflow is: Search companies → check details → analyze risks → analyze opportunities → generate report"
    )

    print("\nAvailable investment thresholds:")
    print("- 70 billion")
    print("- 80 billion")
    print("- 90 billion")
    print("- 100 billion")

    # Run the agent
    while True:
        min_investment = input(
            "\nEnter an investment threshold to start the workflow (or 'quit' to exit): "
        )
        if min_investment.lower() == "quit":
            break

        run_hardcoded_workflow(min_investment, agent)


if __name__ == "__main__":
    main()
