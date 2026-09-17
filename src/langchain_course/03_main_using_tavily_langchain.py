from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from pychomsky.chchat import GCPVertexAnthropicChatWrapper
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

load_dotenv()

tavily_search = TavilySearch(max_results=10)

llm = GCPVertexAnthropicChatWrapper(
    chgw_endpoint="https://chomskygw.vip.qa.ebay.com/api/v1/genai",
    model_name=("gcp-chat-completions-anthropic-claude-sonnet-4.6-sandbox"),
    temperature=0,
    max_output_tokens=1000,
    debug=True,
)
toolList = [tavily_search]
agent = create_agent(
    model = llm,
    tools = toolList)

def main() -> None:
    print("Hello from langchain-course!")
    response = agent.invoke({"messages": [HumanMessage(content=
                                                       """ search for 3 job postings for an ai engineer
                                                       using langchain in the bay area on linkedin and
                                                       list their details"""
                                                       )
                                          ]
                             }
                            )
    print_response(response)

def print_response(response: dict) -> None:
    for m in response["messages"]:
        m.pretty_print()
        print("\n")

if __name__ == "__main__":
    main()