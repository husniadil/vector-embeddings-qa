import os
import json
import time
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
        "description": "High-performance laptop optimized for AI development with NVIDIA GPU",
    },
    "smart-assistant": {
        "name": "Smart Home Assistant",
        "price": 129.99,
        "category": "Smart Home",
        "stock": 45,
        "description": "Voice-controlled smart assistant with advanced AI capabilities",
    },
    "ai-camera": {
        "name": "AI-Powered Security Camera",
        "price": 199.99,
        "category": "Security",
        "stock": 28,
        "description": "Security camera with facial recognition and anomaly detection",
    },
    "ml-toolkit": {
        "name": "Machine Learning Developer Toolkit",
        "price": 299.99,
        "category": "Software",
        "stock": 0,
        "description": "Comprehensive software suite for machine learning development",
    },
    "neural-headphones": {
        "name": "Neural Adaptive Headphones",
        "price": 249.99,
        "category": "Audio",
        "stock": 12,
        "description": "Headphones that adapt to your listening preferences using neural networks",
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


def run_hardcoded_workflow(category, agent):
    """Run a hardcoded workflow with predefined steps:
    1. Search for products in a category
    2. Get details for a specific product
    3. Check product availability
    4. Add product to cart
    5. Process the order
    """
    print(f"\nStep 1: Searching for products in category '{category}'...")
    search_prompt = f"Search for products in the {category} category"
    search_result = agent.run(search_prompt)

    # Extract the product ID directly from the database instead of parsing the response
    matching_products = {}
    for product_id, product in PRODUCT_DATABASE.items():
        if product["category"].lower() == category.lower():
            matching_products[product_id] = product

    if not matching_products:
        print("No products found in this category.")
        return

    # Get the first product ID from the matching products
    product_id = list(matching_products.keys())[0]
    product_name = matching_products[product_id]["name"]

    print(f"Found product: {product_name} (ID: {product_id})")

    print(f"\nStep 2: Getting details for '{product_name}'...")
    details_prompt = f"Get details for product {product_id}"
    details_result = agent.run(details_prompt)

    print(f"\nStep 3: Checking availability for '{product_name}'...")
    availability_prompt = f"Check if {product_id} is available"
    availability_result = agent.run(availability_prompt)

    # Check if the product is available before proceeding
    if "out of stock" in availability_result.content.lower():
        print(f"Product '{product_name}' is out of stock. Workflow terminated.")
        return

    print(f"\nStep 4: Adding '{product_name}' to cart...")
    cart_prompt = f"Add 1 unit of {product_id} to cart"
    cart_result = agent.run(cart_prompt)

    print(f"\nStep 5: Processing order for '{product_name}'...")
    order_prompt = f"Process order for 1 unit of {product_id} for customer@example.com"
    order_result = agent.run(order_prompt)

    print("\nWorkflow completed successfully!")


def create_hardcoded_multi_tool_agent():
    """Create a Hardcoded Multi-Tool Agent

    This agent follows a predefined sequence of steps (static flow).
    Suitable for simple automation processes (RPA style).
    Example: "Search product → check stock → add to cart → process order."
    """
    agent = Agent(
        name="E-commerce Process Agent",
        model=OpenAIChat(id="gpt-3.5-turbo"),
        # No knowledge base
        # No memory
        tools=[
            search_products_by_category,
            get_product_details,
            check_product_availability,
            add_to_cart,
            process_order,
        ],
        instructions=[
            "You are an e-commerce process agent that helps customers with their shopping.",
            "You can search for products, check details, verify availability, add items to cart, and process orders.",
            "For each request, use the appropriate tool based on the specific step in the process.",
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
        "The workflow is: Search product → check details → check stock → add to cart → process order"
    )
    print("\nAvailable product categories:")
    print("- Computers")
    print("- Smart Home")
    print("- Security")
    print("- Software")
    print("- Audio")

    # Run the hardcoded workflow
    while True:
        category = input(
            "\nEnter a product category to start the workflow (or 'quit' to exit): "
        )
        if category.lower() == "quit":
            break

        # Run the predefined workflow with the specified category
        run_hardcoded_workflow(category, agent)


if __name__ == "__main__":
    main()
