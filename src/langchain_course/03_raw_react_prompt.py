from dotenv import load_dotenv

load_dotenv()

# CHANGE 1: Add re + inspect — we'll parse tool calls from raw text instead of structured JSON.
import re
import inspect
import ollama
from langsmith import traceable

# MODEL = "gpt-oss:20b"
MODEL = "llama3.1:latest"
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


@traceable(name="apply_discount", run_type="tool")
def apply_discount(price: float, discount_tier: str) -> str:
    """Apply a discount tier to the product and return the discounted final price of the product.
    Available discount_tier are bronze, silver and gold
    """
    print(
        f"    >> Executing tool: apply_discount: Applying discount {discount_tier} to price {price}"
    )
    discount_percentages = {"bronze": 5, "silver": 10, "gold": 20}
    price = float(price)
    discount = discount_percentages.get(discount_tier, 0)
    final_price = round(price * (1 - discount / 100), 2)
    return str(final_price)


tools_dict = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}


# CHANGE 2: Delete the JSON schemas. Tools now live inside the prompt as plain text.
# Helper function to get the tool descriptions from the tool functions. This will be a
# one big string with the tool name, signature and description that will be fed into the LLM.
def get_tool_description(tools_dict: dict) -> str:
    descriptions = []
    for tool_name, tool in tools_dict.items():
        tool_signature = inspect.signature(tool)
        tool_doc = inspect.getdoc(tool)
        descriptions.append(f"{tool_name}{tool_signature} - {tool_doc}")
    concatenated_tool_descriptions = "\n".join(descriptions)
    print(f"Tool descriptions: {concatenated_tool_descriptions}")
    return concatenated_tool_descriptions


tool_descriptions = get_tool_description(tools_dict)
tool_names = ", ".join(tools_dict.keys())

react_prompt = f"""
You are a helpful assistant that can answer questions about the product catalog
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

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:
"""


# CHANGE 4: Drop tools= from ollama.chat(). The LLM has no idea it's an agent —
# all agency comes from the prompt above and our regex parsing below.
@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages, options):
    return ollama.chat(model=MODEL, messages=messages, options=options)


# --- Agent Loop ---
@traceable(name="Ollama Agent Loop")
def run_agent_loop(question: str):
    print(f"Question: {question}")
    print("=" * 60)

    # CHANGE 5: One prompt string replaces the system/user message split.
    prompt = react_prompt.format(question=question)
    scratchpad = ""

    # ----- Agent Loop -----
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n-----Iteration {iteration} -----")
        full_prompt = prompt + scratchpad
        print(f"Full Prompt:\n{full_prompt}")

        # Stop token prevents the LLM from generating its own Observation —
        # we inject the real tool result instead.
        response = ollama_chat_traced(
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation"], "temperature": 0},
        )
        output = response.message.content
        print("." * 60)
        print(f"LLM Output:\n{output}")
        print("." * 60)
        
        print(f"  [Parsing] Looking for Final Answer in LLM output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"  [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer

        # CHANGE 6: Parse tool calls from raw text with regex — fragile if LLM doesn't follow format.
        print(f"  [Parsing] Looking for Action and Action Input in LLM output...")

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print("  [Parsing] ERROR: Could not parse Action/Action Input from LLM output")
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"  [Tool Selected] {tool_name} with args: {tool_input_raw}")

        # Split comma-separated args; strip key= prefix if LLM outputs key=value format
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"  [Tool Executing] {tool_name}({args})...")
        if tool_name not in tools_dict:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools_dict.keys())}"
        else:
            observation = str(tools_dict[tool_name](*args))


        print(f"  [Tool Result] {observation}")

        # CHANGE 7: History is one growing string re-sent every iteration (replaces messages.append).
        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent_loop("What is the price of a laptop after applying a gold discount?")