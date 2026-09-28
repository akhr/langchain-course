from dotenv import load_dotenv

load_dotenv()

import ollama
from langsmith import traceable

MODEL = "gpt-oss:20b"
MAX_ITERATIONS = 5


@traceable(name="get_product_price", run_type="tool")
def get_product_price(product: str) -> str:
    """Get the price of a product from the catalog"""
    print(f"    >> Executing tool: get_product_price: Getting price for {product}")
    catalog = {
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 89.50,
        "monitor": 325.0,
    }
    price = catalog.get(product, 0.0)
    return str(price)


@traceable(name="get_product_price", run_type="tool")
def apply_discount(price: float, discount_tier: str) -> str:
    """Apply a discount tier to the product and return the discounted final price of the product.
    Available discount_tier are bronze, silver and gold
    """
    print(
        f"    >> Executing tool: apply_discount: Applying discount {discount_tier} to price {price}"
    )
    discount_percentages = {"bronze": 5, "silver": 10, "gold": 20}
    discount = discount_percentages.get(discount_tier, 0)
    final_price = round(price * (1 - discount / 100), 2)
    return str(final_price)


# Difference 2: Without @tool, we must MANUALLY define the JSON schema for each function.
# This is exactly what LangChain's @tool decorator generates automatically
# from the function's type hints and docstring.
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

# NOTE: Ollama can also auto-generate these schemas if you pass the functions
# directly as tools (similar to LangChain's @tool decorator):
#   tools_for_llm = [get_product_price, apply_discount]
# However, this requires your docstrings to follow the Google docstring format
# so Ollama can parse parameter descriptions from the Args section. For example:
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# We keep the manual JSON version here so you can see what @tool hides from you.


# --- Helper: traced Ollama call ---
# Difference 3: Without LangChain, we must manually trace LLM calls for LangSmith.
@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


# --- Agent Loop ---
@traceable(name="Ollama Agent Loop")
def run_agent_loop(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant that can answer questions about the product catalog
                    and apply discounts to the products. You have access to the following tools:
                    - get_product_price: to get the price of a product
                    - apply_discount: to apply a discount to the product
                    STRICTLY the below instructions:
                    - Call apply_discount tool only after getting the price from get_product_price tool.
                    - Do not skip tools. Always call the tools in the order they are provided in the system message.
                    - Pass the exact price from get_product_price tool to the apply_discount tool.
                    - Do not guess product price or discount tier. Always use the tools to answer the question.
                    - Never calculate the price or discount tier yourself. Always use the tools to answer the question.
                    - If the question is not related to the product catalog, say you don't know
                    - If the question is not related to the tools, say you don't know
                    - If product asked by the user is not in the catalog, say product not found
                    - If the discount tier is not in the available discount tiers, say invalid discount tier
                    - If the price is not a number, say invalid price. Ask user to provide a valid price.
                    - Always provide the final price in the response.
                    - When calling tools, always use the exact tool name and arguments as provided in the tool description.
                """,
        },
        {"role": "user", "content": question},
    ]

    # ----- Agent Loop -----
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n-----Iteration {iteration} -----")

        # Invoke the LLM with the messages
        print(f"Iteration {iteration} - messages sent to LLM: ", messages)
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content

        # Process only the FIRST tool call — force one tool per iteration
        tool_call = tool_calls[0]
        # Difference 6: Attribute access (.function.name) instead of dict access (.get("name"))
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        # Get the tool to use and execute the tool
        tool_to_use = tools_dict[tool_name]
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        # Execute the tool and get the observation
        observation = tool_to_use(**tool_args)
        print(f"    >> Tool Response: {observation}")

        # Update the messages with llm and tool response
        messages.append(ai_message)
        messages.append({"role": "tool", "content": str(observation)})

        # continue the loop
    print("ERROR: MAX_ITERATION limit reached with no final answer")
    return None


if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools())")
    print("=" * 60)
    run_agent_loop(question="what is the price of a laptop with a bronze discount?")
    print("=" * 60)
