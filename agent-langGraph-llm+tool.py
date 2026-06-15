from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.environ["GROQ_API_KEY"])

# Tool: used only when LLM needs external/current information
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

# ReAct agent: LLM decides on its own whether to call a tool or answer directly
agent = create_react_agent(llm, tools)

messages = []  # keeps full conversation history

while True:
    question = input("You: ").strip()
    if not question:
        continue
    if question.lower() in ("exit", "quit", "q"):
        break
    messages.append(HumanMessage(content=question))
    try:
        response = agent.invoke({"messages": messages})
        messages = response["messages"]  # update history with agent's reply
        print("Agent:", messages[-1].content)
    except Exception as e:
        messages.pop()  # remove failed message so history stays clean
        print("Agent: Sorry, I couldn't process that. Try rephrasing.")
    print()
