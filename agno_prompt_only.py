import os
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
        name="Content Generator",
        model=OpenAIChat(id="gpt-3.5-turbo"),
        # No knowledge base
        # No tools
        # No memory
        instructions=[
            "You are a creative content generator specialized in AI and technology.",
            "Your job is to generate creative and engaging content based on the user's request.",
            "You can generate blog titles, article outlines, product names, slogans, or keywords.",
            "Be creative, concise, and focus on AI technology themes.",
            "For blog titles, create catchy and informative titles.",
            "For article outlines, provide a structured outline with main points.",
            "For product names, suggest innovative and memorable names.",
            "For slogans, create short, catchy phrases that highlight key benefits.",
            "For keywords, provide relevant keywords for SEO optimization.",
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
    print("- 'Generate 5 blog titles about AI investments'")
    print("- 'Create an outline for an article about Big Tech AI spending'")
    print("- 'Suggest 3 product names for an AI-powered investment analysis tool'")
    print("- 'Write a slogan for a company specializing in AI infrastructure'")
    print("- 'Provide 10 keywords for an article about AI investments'")

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
