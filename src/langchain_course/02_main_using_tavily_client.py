from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from pychomsky.chchat import GCPVertexAnthropicChatWrapper
from langchain_ollama import ChatOllama
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()


@tool
def search_web(query: str) -> str:
    """Tool searches the web for the query and returns the results.

    Args:
        query: The query to search for.
    Returns:
        The results of the search.
    """
    print(f"Searching the web for {query}")
    return tavily.search(query=query)


# llm = GCPVertexAnthropicChatWrapper(
#     chgw_endpoint="https://chomskygw.vip.qa.ebay.com/api/v1/genai",
#     model_name=("gcp-chat-completions-anthropic-claude-sonnet-4.6-sandbox"),
#     temperature=0,
#     max_output_tokens=1000,
#     debug=True,
# )
ollama_llm = ChatOllama(model="gpt-oss:20b", temperature=0)
toolList = [search_web]
agent = create_agent(model=ollama_llm, tools=toolList)


def main() -> None:
    print("Hello from langchain-course!")
    response = agent.invoke(
        {"messages": [HumanMessage(content="What is the weather in Tokyo?")]}
    )
    print_response(response)


def print_response(response: dict) -> None:
    for m in response["messages"]:
        m.pretty_print()
        print("\n")


if __name__ == "__main__":
    main()
