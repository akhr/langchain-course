from doctest import debug
from typing import final, reveal_type

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, tool
from langchain_core.tools import BaseTool, tool
from langchain_ollama import OllamaLLM
from langsmith import traceable
from pychomsky.chchat import GCPVertexAnthropicChatWrapper

# MODEL="gcp-chat-completions-anthropic-claude-sonnet-4.6-sandbox"
MODEL = "gpt-oss:20b"
MAX_ITERATIONS = 5


@tool
def get_product_price(product: str) -> float:
    """Get the price of a product from the catalog"""
    print(f"    >> Executing tool: get_product_price: Getting price for {product}")
    catalog = {
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 89.50,
        "monitor": 325.0,
    }
    price = catalog.get(product, 0.0)
    print(f"The price of {product} is {price}")
    return price


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to the product and return the discounted final price of the product.
    Available discount_tier are bronze, silver and gold"""
    print(
        f"    >> Executing tool: apply_discount: Applying discount {discount_tier} to price {price}"
    )
    discount_percentages = {"bronze": 5, "silver": 10, "gold": 20}
    discount = discount_percentages.get(discount_tier, 0)
    final_price = round(price * (1 - discount / 100), 2)
    return final_price


@traceable(name="Langchain Agent Loop")
def run_agent_loop(question: str):
    print(f"Starting the agent loop for the user's question: {question}")
    print("-" * 30)



if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools())")
    print("-" * 30)
    run_agent_loop("what is the price of a laptop with a gold discount?")
