from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langsmith import traceable

# MODEL="gcp-chat-completions-anthropic-claude-sonnet-4.6-sandbox"
MODEL = "gpt-oss:20b"
# MODEL = "llama3.1:latest"
# MODEL = "gemma3:270m" - Does not support tool calling
MAX_ITERATIONS = 5


@tool
def get_product_price(product: str) -> str:
    """Get the price of a product from the catalog
    Args:
        product: str - The product to get the price of
    Returns:
        float - The price of the product
    """
    print(f"    >> Executing tool: get_product_price: Getting price for {product}")
    catalog = {
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 89.50,
        "monitor": 325.0,
    }
    price = catalog.get(product, 0.0)
    return str(price)


@tool
def apply_discount(price: float, discount_tier: str) -> str:
    """Apply a discount tier to the product and return the discounted final price of the product.
    Available discount_tier are bronze, silver and gold
    Args:
        price: float - The price of the product
        discount_tier: str - The discount tier to apply
    Returns:
        The discounted final price of the product
    """
    print(
        f"    >> Executing tool: apply_discount: Applying discount {discount_tier} to price {price}"
    )
    discount_percentages = {"bronze": 5, "silver": 10, "gold": 20}
    discount = discount_percentages.get(discount_tier, 0)
    final_price = round(price * (1 - discount / 100), 2)
    return str(final_price)


@traceable(name="Langchain Agent Loop")
def run_agent_loop(question: str):
    # Define the tools and tools dictionary
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    messages = [
        SystemMessage(
            content="""You are a helpful assistant that can answer questions about the product catalog 
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
                """
        ),
        HumanMessage(content=question),
    ]

    # Get the LLM and bind the tools
    llm = get_llm(tools)
    llm_with_tools = llm.bind_tools(tools=tools)

    # Agent Loop 
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n-----Iteration {iteration} -----")

        # Invoke the LLM with the messages
        ai_message = llm_with_tools.invoke(input=messages)
        # print(f"LLM Response: {ai_message}")

        # If the LLM response does not have tool calls, return the content
        if not ai_message.tool_calls:
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content

        # If the LLM response has tool calls, get the tool call and execute the tool
        tool_call = ai_message.tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args")
        tool_call_id = tool_call.get("id")

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        # Get the tool to use and execute the tool
        tool_to_use = tools_dict[tool_name]
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")

        # Execute the tool and get the observation
        observation = tool_to_use.invoke(input=tool_args)
        print(f"    >> Tool Response: {observation}")

        # Update the messages with llm and tool response
        messages.append(ai_message)
        messages.append(ToolMessage(content=observation, tool_call_id=tool_call_id))

        # continue the loop
    print("ERROR: MAX_ITERATION limit reached with no final answer")
    return None


def get_llm(tools: list[BaseTool]) -> BaseChatModel:
    llm = init_chat_model(model=f"ollama:{MODEL}", temperature=0.0)
    return llm


if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools())")
    print("=" * 60)
    run_agent_loop(question= "what is the price of a laptop with a gold discount?")
    print("=" * 60)
