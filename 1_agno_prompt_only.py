from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat

# Load environment variables
load_dotenv()


def create_prompt_only_agent():
    """Create a Prompt-Only Agent (Stateless LLM)

    This agent only relies on prompt and response from the LLM.
    No tool calls, memory, or planning.
    Suitable for: content generation, naming, keyword suggestion, etc.
    """
    agent = Agent(
        name="AI Investment Content Creator",
        model=OpenAIChat(id="gpt-4.1-mini"),
        # No knowledge base
        # No tools
        # No memory
        instructions=[
            "You are a creative content creator specializing in AI investment topics.",
            "Your job is to generate creative and engaging content based on the user's request.",
            "You can generate blog titles, article outlines, investment slogans, or keywords related to AI investments.",
            "Be creative, concise, and focus on Big Tech AI investment themes.",
            "For blog titles, create catchy and informative titles about AI investments.",
            "For article outlines, provide a structured outline with main points about AI investment topics.",
            "For investment slogans, create short, catchy phrases that highlight the potential of AI investments.",
            "For keywords, provide relevant keywords for SEO optimization related to AI investments.",
            "You have knowledge about major tech companies like Microsoft, Google, Amazon, Meta, and their AI investments.",
            "You know that Big Tech companies are planning to spend over $320 billion on AI in 2025.",
            "You are aware that Microsoft plans to spend $80 billion, Google $75 billion, and Amazon over $100 billion on AI in 2025.",
            "You understand investor concerns about the massive AI spending and potential ROI challenges.",
            "You know about the impact of events like DeepSeek's R1 model release on the tech market.",
            "You're familiar with the 'Magnificent Seven' tech companies and their dominance in AI investments.",
        ],
        markdown=True,
    )

    return agent


def main():
    print("Creating a Prompt-Only Agent (Stateless LLM)...")
    agent = create_prompt_only_agent()

    print("\n=== Prompt-Only Agent Demo ===")
    print(
        "This demo shows a simple agent that only relies on prompt and response from the LLM."
    )
    print("No tool calls, memory, or planning.")
    print("Suitable for: content generation, naming, keyword suggestion, etc.")
    print("\nExample requests:")
    print("- 'Generate 5 blog titles about Big Tech AI investments'")
    print(
        "- 'Create an outline for an article about the ROI challenges of AI spending'"
    )
    print("- 'Suggest 3 catchy slogans for an AI investment newsletter'")
    print(
        "- 'Write a short paragraph comparing Microsoft and Google's AI investment strategies'"
    )
    print("- 'Provide 10 keywords for an article about AI infrastructure investments'")

    # Run the agent
    while True:
        request = input("\nEnter your request (or 'quit' to exit): ")
        if request.lower() == "quit":
            break

        print(f"\nGenerating content for: '{request}'...")
        # Use the agent to generate content
        agent.print_response(request, stream=True)


if __name__ == "__main__":
    main()
