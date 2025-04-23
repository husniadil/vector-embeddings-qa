import os
import json
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool

# Load environment variables
load_dotenv()


# Define a simple product database
PRODUCT_DATABASE = {
    "laptop-ai-pro": {
        "name": "AI Pro Laptop",
        "price": 1299.99,
        "category": "Computers",
        "stock": 15,
        "description": "High-performance laptop optimized for AI development with NVIDIA GPU"
    },
    "smart-assistant": {
        "name": "Smart Home Assistant",
        "price": 129.99,
        "category": "Smart Home",
        "stock": 45,
        "description": "Voice-controlled smart assistant with advanced AI capabilities"
    },
    "ai-camera": {
        "name": "AI-Powered Security Camera",
        "price": 199.99,
        "category": "Security",
        "stock": 28,
        "description": "Security camera with facial recognition and anomaly detection"
    },
    "ml-toolkit": {
        "name": "Machine Learning Developer Toolkit",
        "price": 299.99,
        "category": "Software",
        "stock": 0,
        "description": "Comprehensive software suite for machine learning development"
    },
    "neural-headphones": {
        "name": "Neural Adaptive Headphones",
        "price": 249.99,
        "category": "Audio",
        "stock": 12,
        "description": "Headphones that adapt to your listening preferences using neural networks"
    }
}


@tool
def check_product_stock(product_id: str) -> str:
    """Check the stock availability of a product.
    
    Args:
        product_id: The ID of the product to check
        
    Returns:
        Information about the product stock
    """
    product_id = product_id.lower().strip()
    
    if product_id in PRODUCT_DATABASE:
        product = PRODUCT_DATABASE[product_id]
        stock = product["stock"]
        
        if stock > 0:
            return f"Product '{product['name']}' is in stock. Available quantity: {stock}."
        else:
            return f"Product '{product['name']}' is currently out of stock."
    else:
        return f"Product with ID '{product_id}' not found in the database."


@tool
def get_product_info(product_id: str) -> str:
    """Get detailed information about a product.
    
    Args:
        product_id: The ID of the product to look up
        
    Returns:
        Detailed information about the product
    """
    product_id = product_id.lower().strip()
    
    if product_id in PRODUCT_DATABASE:
        product = PRODUCT_DATABASE[product_id]
        return json.dumps(product, indent=2)
    else:
        return f"Product with ID '{product_id}' not found in the database."


@tool
def search_products_by_category(category: str) -> str:
    """Search for products in a specific category.
    
    Args:
        category: The category to search for
        
    Returns:
        List of products in the specified category
    """
    category = category.lower().strip()
    
    matching_products = {}
    for product_id, product in PRODUCT_DATABASE.items():
        if product["category"].lower() == category:
            matching_products[product_id] = product
    
    if matching_products:
        return json.dumps(matching_products, indent=2)
    else:
        return f"No products found in category '{category}'."


def create_tool_calling_agent():
    """Create a Level 1 agent - Tool-Calling Agent (Single-Step)
    
    This agent can call a single tool (API, DB, etc.) in one step.
    No planning or context memory.
    Example: "Check stock for product X."
    """
    agent = Agent(
        name="Product Information Agent",
        model=OpenAIChat(id="gpt-3.5-turbo"),
        # No knowledge base
        # No memory
        tools=[check_product_stock, get_product_info, search_products_by_category],
        instructions=[
            "You are a product information assistant for an AI technology store.",
            "You can help customers check product stock, get product information, and search for products by category.",
            "Use the appropriate tool based on the customer's request:",
            "- Use check_product_stock to check if a product is available and how many are in stock",
            "- Use get_product_info to get detailed information about a specific product",
            "- Use search_products_by_category to find products in a specific category",
            "You can only use one tool per request - no multi-step processes.",
            "If a customer asks about a product, try to determine the product ID from our database:",
            "- laptop-ai-pro: AI Pro Laptop",
            "- smart-assistant: Smart Home Assistant",
            "- ai-camera: AI-Powered Security Camera",
            "- ml-toolkit: Machine Learning Developer Toolkit",
            "- neural-headphones: Neural Adaptive Headphones",
            "If the customer doesn't specify a product ID, ask them to clarify which product they're interested in.",
        ],
        markdown=True,
        show_tool_calls=True,
    )
    
    return agent


def main():
    print("Creating a Level 1 Agent - Tool-Calling Agent (Single-Step)...")
    agent = create_tool_calling_agent()

    print("\n=== Level 1: Tool-Calling Agent Demo ===")
    print("This demo shows an agent that can call a single tool in one step.")
    print("No planning or context memory.")
    print("Suitable for: simple information retrieval, checking status, etc.")
    print("\nExample requests:")
    print("- 'Check stock for the AI Pro Laptop'")
    print("- 'Tell me about the Smart Home Assistant'")
    print("- 'What products do you have in the Audio category?'")
    print("- 'Is the Machine Learning Developer Toolkit available?'")
    print("- 'Give me details about the neural headphones'")

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
