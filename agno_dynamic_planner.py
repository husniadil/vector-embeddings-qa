import os
import json
import time
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool
from agno.tools.reasoning import ReasoningTools

# Load environment variables
load_dotenv()


# Define a simple product database
PRODUCT_DATABASE = {
    "laptop-ai-pro": {
        "name": "AI Pro Laptop",
        "price": 1299.99,
        "category": "Computers",
        "stock": 15,
        "description": "High-performance laptop optimized for AI development with NVIDIA GPU",
        "features": ["16GB RAM", "1TB SSD", "NVIDIA RTX 4070", "15.6 inch display"],
        "rating": 4.7,
        "reviews": 128,
    },
    "smart-assistant": {
        "name": "Smart Home Assistant",
        "price": 129.99,
        "category": "Smart Home",
        "stock": 45,
        "description": "Voice-controlled smart assistant with advanced AI capabilities",
        "features": [
            "Voice recognition",
            "Smart home integration",
            "AI-powered responses",
            "Music streaming",
        ],
        "rating": 4.5,
        "reviews": 302,
    },
    "ai-camera": {
        "name": "AI-Powered Security Camera",
        "price": 199.99,
        "category": "Security",
        "stock": 28,
        "description": "Security camera with facial recognition and anomaly detection",
        "features": [
            "Facial recognition",
            "Night vision",
            "Motion detection",
            "Cloud storage",
        ],
        "rating": 4.3,
        "reviews": 89,
    },
    "ml-toolkit": {
        "name": "Machine Learning Developer Toolkit",
        "price": 299.99,
        "category": "Software",
        "stock": 0,
        "description": "Comprehensive software suite for machine learning development",
        "features": [
            "Model training",
            "Data preprocessing",
            "Visualization tools",
            "API integration",
        ],
        "rating": 4.8,
        "reviews": 56,
    },
    "neural-headphones": {
        "name": "Neural Adaptive Headphones",
        "price": 249.99,
        "category": "Audio",
        "stock": 12,
        "description": "Headphones that adapt to your listening preferences using neural networks",
        "features": [
            "Noise cancellation",
            "Adaptive EQ",
            "Wireless",
            "30-hour battery",
        ],
        "rating": 4.6,
        "reviews": 175,
    },
}

# Define customer profiles for personalization
CUSTOMER_PROFILES = {
    "developer": {
        "interests": ["Computers", "Software"],
        "budget": 2000,
        "preferences": ["High performance", "Development tools"],
    },
    "home_user": {
        "interests": ["Smart Home", "Audio"],
        "budget": 500,
        "preferences": ["Ease of use", "Integration"],
    },
    "security_conscious": {
        "interests": ["Security"],
        "budget": 1000,
        "preferences": ["Reliability", "Advanced features"],
    },
}

# Define a simple order database
ORDER_DATABASE = {}


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


@tool
def search_products_by_price_range(min_price: float, max_price: float) -> str:
    """Search for products within a specific price range.

    Args:
        min_price: The minimum price
        max_price: The maximum price

    Returns:
        List of products within the price range
    """
    matching_products = {}
    for product_id, product in PRODUCT_DATABASE.items():
        if min_price <= product["price"] <= max_price:
            matching_products[product_id] = product

    if matching_products:
        return json.dumps(matching_products, indent=2)
    else:
        return f"No products found within price range ${min_price} - ${max_price}."


@tool
def search_products_by_feature(feature: str) -> str:
    """Search for products with a specific feature.

    Args:
        feature: The feature to search for

    Returns:
        List of products with the specified feature
    """
    feature = feature.lower().strip()

    matching_products = {}
    for product_id, product in PRODUCT_DATABASE.items():
        for prod_feature in product["features"]:
            if feature in prod_feature.lower():
                matching_products[product_id] = product
                break

    if matching_products:
        return json.dumps(matching_products, indent=2)
    else:
        return f"No products found with feature '{feature}'."


@tool
def get_product_details(product_id: str) -> str:
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
def check_product_availability(product_id: str) -> str:
    """Check if a product is available in stock.

    Args:
        product_id: The ID of the product to check

    Returns:
        Availability status of the product
    """
    product_id = product_id.lower().strip()

    if product_id in PRODUCT_DATABASE:
        product = PRODUCT_DATABASE[product_id]
        stock = product["stock"]

        if stock > 0:
            return (
                f"Product '{product['name']}' is in stock. Available quantity: {stock}."
            )
        else:
            return f"Product '{product['name']}' is currently out of stock."
    else:
        return f"Product with ID '{product_id}' not found in the database."


@tool
def compare_products(product_id1: str, product_id2: str) -> str:
    """Compare two products side by side.

    Args:
        product_id1: The ID of the first product
        product_id2: The ID of the second product

    Returns:
        Comparison of the two products
    """
    product_id1 = product_id1.lower().strip()
    product_id2 = product_id2.lower().strip()

    if product_id1 not in PRODUCT_DATABASE:
        return f"Product with ID '{product_id1}' not found in the database."

    if product_id2 not in PRODUCT_DATABASE:
        return f"Product with ID '{product_id2}' not found in the database."

    product1 = PRODUCT_DATABASE[product_id1]
    product2 = PRODUCT_DATABASE[product_id2]

    comparison = {
        "comparison": {
            "name": [product1["name"], product2["name"]],
            "price": [product1["price"], product2["price"]],
            "category": [product1["category"], product2["category"]],
            "stock": [product1["stock"], product2["stock"]],
            "rating": [product1["rating"], product2["rating"]],
            "reviews": [product1["reviews"], product2["reviews"]],
            "features": [product1["features"], product2["features"]],
        }
    }

    return json.dumps(comparison, indent=2)


@tool
def get_customer_profile(profile_type: str) -> str:
    """Get a customer profile for personalized recommendations.

    Args:
        profile_type: The type of customer profile (developer, home_user, security_conscious)

    Returns:
        Customer profile information
    """
    profile_type = profile_type.lower().strip()

    if profile_type in CUSTOMER_PROFILES:
        return json.dumps(CUSTOMER_PROFILES[profile_type], indent=2)
    else:
        return f"Customer profile '{profile_type}' not found. Available profiles: {', '.join(CUSTOMER_PROFILES.keys())}"


@tool
def generate_personalized_recommendations(profile_type: str) -> str:
    """Generate personalized product recommendations based on a customer profile.

    Args:
        profile_type: The type of customer profile (developer, home_user, security_conscious)

    Returns:
        Personalized product recommendations
    """
    profile_type = profile_type.lower().strip()

    if profile_type not in CUSTOMER_PROFILES:
        return f"Customer profile '{profile_type}' not found. Available profiles: {', '.join(CUSTOMER_PROFILES.keys())}"

    profile = CUSTOMER_PROFILES[profile_type]

    # Find products that match the profile's interests and are within budget
    recommendations = {}
    for product_id, product in PRODUCT_DATABASE.items():
        if (
            product["category"] in profile["interests"]
            and product["price"] <= profile["budget"]
        ):
            recommendations[product_id] = product

    if recommendations:
        return json.dumps(recommendations, indent=2)
    else:
        return f"No suitable recommendations found for profile '{profile_type}'."


@tool
def add_to_cart(product_id: str, quantity: int) -> str:
    """Add a product to the shopping cart.

    Args:
        product_id: The ID of the product to add
        quantity: The quantity to add

    Returns:
        Confirmation message
    """
    product_id = product_id.lower().strip()

    if product_id in PRODUCT_DATABASE:
        product = PRODUCT_DATABASE[product_id]
        available_stock = product["stock"]

        if available_stock <= 0:
            return f"Cannot add to cart. Product '{product['name']}' is out of stock."

        if quantity > available_stock:
            return f"Cannot add {quantity} units to cart. Only {available_stock} units of '{product['name']}' are available."

        # In a real application, this would update a user's cart
        # For this demo, we'll just return a success message
        return f"Successfully added {quantity} unit(s) of '{product['name']}' to cart."
    else:
        return f"Product with ID '{product_id}' not found in the database."


@tool
def process_order(product_id: str, quantity: int, customer_email: str) -> str:
    """Process an order for a product.

    Args:
        product_id: The ID of the product to order
        quantity: The quantity to order
        customer_email: The customer's email address

    Returns:
        Order confirmation
    """
    product_id = product_id.lower().strip()

    if product_id in PRODUCT_DATABASE:
        product = PRODUCT_DATABASE[product_id]
        available_stock = product["stock"]

        if available_stock <= 0:
            return f"Cannot process order. Product '{product['name']}' is out of stock."

        if quantity > available_stock:
            return f"Cannot order {quantity} units. Only {available_stock} units of '{product['name']}' are available."

        # Generate a simple order ID
        order_id = f"ORD-{int(time.time())}"

        # Update stock (in a real application, this would be a database update)
        PRODUCT_DATABASE[product_id]["stock"] -= quantity

        # Save order to database
        ORDER_DATABASE[order_id] = {
            "product_id": product_id,
            "product_name": product["name"],
            "quantity": quantity,
            "customer_email": customer_email,
            "total_price": product["price"] * quantity,
            "status": "confirmed",
        }

        return f"Order confirmed! Order ID: {order_id}. {quantity} unit(s) of '{product['name']}' will be shipped to {customer_email}. Total: ${product['price'] * quantity:.2f}"
    else:
        return f"Product with ID '{product_id}' not found in the database."


def create_dynamic_planner_agent():
    """Create a Dynamic Planner Agent

    This agent can dynamically plan steps based on the goal.
    It can determine the order of tool calls on its own.
    Example: product recommendation with automatic breakdown.
    """
    agent = Agent(
        name="AI Shopping Assistant",
        model=OpenAIChat(id="gpt-3.5-turbo"),
        # No knowledge base
        # No persistent memory
        tools=[
            ReasoningTools(),
            search_products_by_category,
            search_products_by_price_range,
            search_products_by_feature,
            get_product_details,
            check_product_availability,
            compare_products,
            get_customer_profile,
            generate_personalized_recommendations,
            add_to_cart,
            process_order,
        ],
        instructions=[
            "You are an AI shopping assistant that helps customers find and purchase AI technology products.",
            "You can dynamically plan and execute steps based on the customer's goal.",
            "First, understand what the customer wants to achieve.",
            "Then, create a plan with the necessary steps to fulfill their request.",
            "Execute the plan by calling the appropriate tools in the right order.",
            "You can search for products by category, price range, or features.",
            "You can get detailed information about products and check their availability.",
            "You can compare products side by side to help customers make decisions.",
            "You can generate personalized recommendations based on customer profiles.",
            "You can add products to cart and process orders.",
            "Be proactive and helpful, suggesting additional steps that might benefit the customer.",
            "Always explain your reasoning and plan before taking actions.",
        ],
        markdown=True,
        show_tool_calls=True,
    )

    return agent


def main():
    print("Creating a Dynamic Planner Agent...")
    agent = create_dynamic_planner_agent()

    print("\n=== Dynamic Planner Agent Demo ===")
    print("This demo shows an agent that can dynamically plan steps based on the goal.")
    print("The agent determines the order of tool calls on its own.")
    print("Suitable for: complex tasks requiring reasoning and multi-step planning.")
    print("\nExample requests:")
    print("- 'Find me the best AI development laptop'")
    print("- 'I need a smart home product under $150'")
    print("- 'Compare the AI Pro Laptop with the Neural Adaptive Headphones'")
    print("- 'Recommend products for a software developer'")
    print("- 'I want to buy a security camera with facial recognition'")

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
