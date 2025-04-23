import os
import json
import time
import datetime
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.reasoning import ReasoningTools
from agno.tools import tool
from agno.storage.sqlite import SqliteStorage
from agno.knowledge.text import TextKnowledgeBase
from agno.vectordb.lancedb import LanceDb
from agno.embedder.openai import OpenAIEmbedder
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.memory import Memory

# Load environment variables
load_dotenv()

completion_model = os.getenv("COMPLETION_MODEL", "gpt-4.1-mini")

# Define a database of Big Tech companies and their AI investments
COMPANY_DATABASE = {
    "microsoft": {
        "name": "Microsoft Corporation",
        "ticker": "MSFT",
        "ai_investments": {
            "openai": {
                "amount": "$13 billion",
                "year": 2023,
                "details": "Strategic partnership with OpenAI, creator of ChatGPT"
            },
            "anthropic": {
                "amount": "$100 million",
                "year": 2023,
                "details": "Investment in Anthropic, creator of Claude AI assistant"
            },
            "internal": {
                "amount": "$80 billion",
                "year": 2025,
                "details": "Planned investment in AI research and development"
            }
        },
        "ai_products": [
            "Microsoft Copilot",
            "Azure OpenAI Service",
            "Bing Chat",
            "GitHub Copilot"
        ],
        "ai_strategy": "Microsoft is focusing on integrating AI across its product suite, with a particular emphasis on productivity tools and cloud services. Their partnership with OpenAI has been central to their AI strategy."
    },
    "alphabet": {
        "name": "Alphabet Inc. (Google)",
        "ticker": "GOOGL",
        "ai_investments": {
            "anthropic": {
                "amount": "$300 million",
                "year": 2023,
                "details": "Investment in Anthropic, creator of Claude AI assistant"
            },
            "internal": {
                "amount": "$75 billion",
                "year": 2025,
                "details": "Planned investment in AI research and development"
            }
        },
        "ai_products": [
            "Google Bard",
            "Google Cloud AI",
            "Google Search AI",
            "DeepMind"
        ],
        "ai_strategy": "Google is leveraging its vast data resources and research capabilities to develop AI solutions across search, cloud, and consumer products. DeepMind remains their primary AI research arm."
    },
    "amazon": {
        "name": "Amazon.com, Inc.",
        "ticker": "AMZN",
        "ai_investments": {
            "anthropic": {
                "amount": "$4 billion",
                "year": 2023,
                "details": "Major investment in Anthropic, creator of Claude AI assistant"
            },
            "internal": {
                "amount": "$100 billion",
                "year": 2025,
                "details": "Planned investment in AI research and development"
            }
        },
        "ai_products": [
            "Amazon Bedrock",
            "AWS AI Services",
            "Alexa",
            "Amazon Q"
        ],
        "ai_strategy": "Amazon is focusing on AI applications in e-commerce, cloud services, and voice assistants. Their AWS division is central to their AI strategy, offering a range of AI services to customers."
    },
    "meta": {
        "name": "Meta Platforms, Inc.",
        "ticker": "META",
        "ai_investments": {
            "llama": {
                "amount": "$5 billion",
                "year": 2023,
                "details": "Development of open-source LLaMA models"
            },
            "internal": {
                "amount": "$35 billion",
                "year": 2025,
                "details": "Planned investment in AI research and development"
            }
        },
        "ai_products": [
            "LLaMA",
            "Meta AI",
            "AI in Instagram and Facebook",
            "AI for AR/VR"
        ],
        "ai_strategy": "Meta is investing heavily in open-source AI models and AI for social media and metaverse applications. Their LLaMA models have gained significant attention in the AI community."
    }
}

# Define market data for the companies
MARKET_DATA = {
    "microsoft": {
        "price": 420.35,
        "market_cap": "3.12T",
        "pe_ratio": 36.5,
        "daily_change": 1.2,
        "ai_revenue_percentage": 15.3,
        "ai_growth_rate": 45.2
    },
    "alphabet": {
        "price": 175.98,
        "market_cap": "2.21T",
        "pe_ratio": 25.1,
        "daily_change": 0.8,
        "ai_revenue_percentage": 12.7,
        "ai_growth_rate": 38.5
    },
    "amazon": {
        "price": 185.45,
        "market_cap": "1.92T",
        "pe_ratio": 42.3,
        "daily_change": -0.5,
        "ai_revenue_percentage": 8.5,
        "ai_growth_rate": 52.1
    },
    "meta": {
        "price": 510.92,
        "market_cap": "1.30T",
        "pe_ratio": 29.7,
        "daily_change": 2.1,
        "ai_revenue_percentage": 6.2,
        "ai_growth_rate": 41.8
    }
}

# Define user preferences
USER_PREFERENCES = {
    "risk_tolerance": "moderate",
    "investment_horizon": "long-term",
    "preferred_companies": ["microsoft", "amazon"],
    "investment_amount": 10000,
    "notification_threshold": 5.0,  # percentage change to trigger notification
}

# Database for storing recommendations
recommendation_db_file = "sqlite_data/recommendations.db"

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
        return f"Company '{company_name}' not found in the database."


@tool
def get_market_data(company_name: str) -> str:
    """Get the latest market data for a company.

    Args:
        company_name: The name of the company to look up

    Returns:
        Latest market data for the company
    """
    company_name = company_name.lower().strip()

    # Handle common variations
    if company_name == "google":
        company_name = "alphabet"
    elif company_name == "facebook":
        company_name = "meta"

    if company_name in MARKET_DATA:
        # Simulate daily change
        MARKET_DATA[company_name]["daily_change"] = round(
            (2 * (time.time() % 10) / 10 - 1) * 3, 2
        )  # Random value between -3% and +3%
        return json.dumps(MARKET_DATA[company_name], indent=2)
    else:
        return f"Market data for company '{company_name}' not found."


@tool
def get_user_preferences() -> str:
    """Get the user's investment preferences.

    Returns:
        User's investment preferences
    """
    return json.dumps(USER_PREFERENCES, indent=2)


@tool
def save_recommendation(company_name: str, recommendation: str, confidence: float) -> str:
    """Save an investment recommendation to the agent's memory.

    Args:
        company_name: The name of the company
        recommendation: The recommendation (Buy, Hold, Sell)
        confidence: Confidence level (0-100)

    Returns:
        Confirmation of saving the recommendation
    """
    # Format the recommendation data
    timestamp = datetime.datetime.now().isoformat()
    
    # Ensure the sqlite_data directory exists
    os.makedirs("sqlite_data", exist_ok=True)
    
    # Store in SQLite database for persistence
    import sqlite3
    conn = sqlite3.connect(recommendation_db_file)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS recommendations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        company TEXT,
        recommendation TEXT,
        confidence REAL,
        timestamp TEXT
    )
    ''')
    
    # Insert the recommendation
    cursor.execute(
        "INSERT INTO recommendations (company, recommendation, confidence, timestamp) VALUES (?, ?, ?, ?)",
        (company_name, recommendation, confidence, timestamp)
    )
    
    conn.commit()
    conn.close()
    
    return f"Recommendation for {company_name} saved: {recommendation} (Confidence: {confidence}%)"


@tool
def get_previous_recommendations(company_name: str = "") -> str:
    """Get previous recommendations from the agent's memory.
    
    Args:
        company_name: Optional company name to filter recommendations
        
    Returns:
        Previous recommendations
    """
    # Ensure the sqlite_data directory exists
    os.makedirs("sqlite_data", exist_ok=True)
    
    # Check if the database file exists
    if not os.path.exists(recommendation_db_file):
        if company_name:
            return f"No previous recommendations found for {company_name}."
        else:
            return "No previous recommendations found in memory."
    
    # Query the SQLite database
    import sqlite3
    conn = sqlite3.connect(recommendation_db_file)
    cursor = conn.cursor()
    
    try:
        # Check if the table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='recommendations'")
        if not cursor.fetchone():
            conn.close()
            if company_name:
                return f"No previous recommendations found for {company_name}."
            else:
                return "No previous recommendations found in memory."
        
        # Query recommendations
        if company_name:
            cursor.execute(
                "SELECT company, recommendation, confidence, timestamp FROM recommendations WHERE LOWER(company) LIKE LOWER(?)",
                (f"%{company_name}%",)
            )
        else:
            cursor.execute("SELECT company, recommendation, confidence, timestamp FROM recommendations ORDER BY timestamp DESC")
        
        recommendations = cursor.fetchall()
        conn.close()
        
        if not recommendations:
            if company_name:
                return f"No previous recommendations found for {company_name}."
            else:
                return "No previous recommendations found in memory."
        
        # Format the results
        result = "Previous recommendations:\n\n"
        for rec in recommendations:
            company, rec_type, confidence, timestamp = rec
            # Parse the timestamp to a more readable format
            try:
                dt = datetime.datetime.fromisoformat(timestamp)
                formatted_date = dt.strftime("%Y-%m-%d %H:%M")
            except:
                formatted_date = timestamp
                
            result += f"- {company}: {rec_type} (Confidence: {confidence}%, Date: {formatted_date})\n"
        
        return result
        
    except Exception as e:
        conn.close()
        return f"Error retrieving recommendations: {str(e)}"


@tool
def create_notification(company_name: str, event_type: str, message: str) -> str:
    """Create a notification for the user.

    Args:
        company_name: The name of the company
        event_type: Type of event (price_change, news, recommendation)
        message: Notification message

    Returns:
        Confirmation of creating the notification
    """
    return f"Notification created for {company_name} ({event_type}): {message}"


def setup_knowledge_base():
    """Set up a knowledge base using LanceDB and the content from content_data/content.txt"""
    # Create the knowledge base with LanceDB as the vector database
    knowledge_base = TextKnowledgeBase(
        path="content_data",  # Use the content_data directory which contains content.txt
        vector_db=LanceDb(
            table_name="embeddings_goal_memory",
            uri="sqlite_data",
            # Use OpenAI embeddings
            embedder=OpenAIEmbedder(
                id=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            ),
        ),
    )

    return knowledge_base


def create_goal_driven_memory_agent():
    """Create a Goal-Driven Agent with Memory

    This agent has memory to store state and previous results.
    It can learn, adapt, and perform recurring tasks.
    Example: daily recommendations, automatic notifications.
    """
    # Set up the knowledge base
    knowledge_base = setup_knowledge_base()

    # Check if we need to load the knowledge base
    # If the vector database already exists, we can skip this step
    if not os.path.exists("sqlite_data/embeddings_goal_memory"):
        print("Loading knowledge base...")
        knowledge_base.load(upsert=True)
        print("Knowledge base loaded successfully!")
    else:
        print("Using existing knowledge base.")

    # Create SQLite storage for the agent
    storage = SqliteStorage(
        table_name="goal_memory_agent",
        db_file="sqlite_data/agent_memory.db",
    )

    # Set up the memory system with SQLite storage
    memory_db = SqliteMemoryDb(
        table_name="investment_memory",
        db_file="sqlite_data/agent_memory.db"
    )
    memory = Memory(db=memory_db)

    # Create the agent with memory
    agent = Agent(
        name="AI Investment Advisor",
        model=OpenAIChat(id=completion_model),
        knowledge=knowledge_base,
        search_knowledge=True,
        # Use Agno's memory system
        memory=memory,
        # Enable memory features
        enable_agentic_memory=True,
        enable_user_memories=True,
        storage=storage,
        # Add history to messages
        add_history_to_messages=True,
        num_history_runs=5,
        tools=[
            ReasoningTools(),
            get_company_info,
            get_market_data,
            get_user_preferences,
            save_recommendation,
            get_previous_recommendations,
            create_notification,
        ],
        instructions=[
            "You are an AI investment advisor specializing in Big Tech AI investments.",
            "You have memory to store state and previous results.",
            "You can learn from past interactions and adapt your recommendations.",
            "You can perform recurring tasks like daily recommendations and notifications.",
            "You track market data and company information to provide timely advice.",
            "You consider user preferences when making recommendations.",
            "You can create notifications for significant events or price changes.",
            "You maintain a history of your recommendations and their performance.",
            "You can compare current data with historical data to identify trends.",
            "You provide personalized recommendations based on user preferences and market conditions.",
            "You focus on Big Tech companies' AI investments and their potential returns.",
            "You know that Big Tech companies are planning to spend over $320 billion on AI in 2025.",
            "You are aware that Microsoft plans to spend $80 billion, Google $75 billion, and Amazon over $100 billion on AI in 2025.",
            "When asked about previous recommendations, use the get_previous_recommendations tool to retrieve them.",
            "IMPORTANT: ALWAYS use the save_recommendation tool when making new recommendations to ensure they are stored in memory.",
            "When generating daily updates, check for significant market changes and create notifications if needed.",
            "Remember user preferences and tailor your recommendations accordingly.",
            "When users tell you their name or preferences, store this information in your memory.",
            "When users ask who they are, retrieve their information from your memory.",
            "Always check previous recommendations before making new ones to ensure consistency.",
            "If a user expresses disinterest in a particular company, remember this preference and avoid recommending it in the future.",
        ],
        markdown=True,
        show_tool_calls=True,
    )

    return agent


def simulate_daily_update(agent, day_number):
    """Simulate a daily update for the agent"""
    today = datetime.datetime.now() + datetime.timedelta(days=day_number)
    date_str = today.strftime("%Y-%m-%d")
    
    print(f"\n=== Daily Update: {date_str} (Day {day_number}) ===")
    
    # Check for significant market changes
    significant_changes = []
    for company_id, data in MARKET_DATA.items():
        # Simulate daily change based on day number for consistency
        change = round(((day_number * 7 + ord(company_id[0])) % 15 - 7) / 2, 2)
        MARKET_DATA[company_id]["daily_change"] = change
        
        if abs(change) >= USER_PREFERENCES["notification_threshold"]:
            significant_changes.append(company_id)
    
    if significant_changes:
        companies_str = ", ".join([COMPANY_DATABASE[c]["name"] for c in significant_changes])
        prompt = f"Today is {date_str}. Generate daily recommendations and notifications. The following companies have significant price changes: {companies_str}."
    else:
        prompt = f"Today is {date_str}. Generate daily recommendations and notifications based on the latest market data."
    
    # Run the agent with the daily update prompt
    response = agent.run(prompt)
    
    # Print the response
    agent.print_response(prompt, stream=True)


def main():
    print("Creating a Goal-Driven Agent with Memory...")
    agent = create_goal_driven_memory_agent()

    print("\n=== Goal-Driven Agent with Memory Demo ===")
    print("This demo shows an agent that has memory to store state and previous results.")
    print("It can learn, adapt, and perform recurring tasks.")
    print("Suitable for: daily recommendations, automatic notifications, etc.")
    
    print("\nExample requests:")
    print("- 'What are your current investment recommendations?'")
    print("- 'Create a daily update for my portfolio'")
    print("- 'Notify me about significant AI investment news'")
    print("- 'How have your recommendations changed over time?'")
    print("- 'Analyze the trend in Microsoft's AI investments'")
    print("- 'My name is [your name]'")
    print("- 'Who am I?'")

    # Run the agent in interactive mode or simulation mode
    mode = input("\nChoose mode (1: Interactive, 2: Simulation): ")
    
    if mode == "2":
        # Simulation mode: Run daily updates for a week
        num_days = int(input("Enter number of days to simulate: "))
        for day in range(1, num_days + 1):
            simulate_daily_update(agent, day)
            if day < num_days:
                input("\nPress Enter to continue to the next day...")
    else:
        # Interactive mode
        while True:
            request = input("\nEnter your request (or 'quit' to exit): ")
            if request.lower() == "quit":
                break

            print(f"\nProcessing request: '{request}'...")
            # Use the agent to process the request
            response = agent.run(request)
            
            # Print the response
            agent.print_response(request, stream=True)


if __name__ == "__main__":
    main()
