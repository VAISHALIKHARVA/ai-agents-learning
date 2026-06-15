from typing import Dict, List
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.environ["GROQ_API_KEY"])

class State(Dict):
    messages: List[Dict[str, str]]

graph_builder = StateGraph(State)

def chatbot(state: State):
    response = llm.invoke(state["messages"])
    state["messages"].append({"role": "assistant", "content": response.content})
    return {"messages": state["messages"]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

app = graph_builder.compile()

if __name__ == "__main__":
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            state = {"messages": [{"role": "user", "content": user_input}]}
            result = app.invoke(state)
            print(f"Assistant: {result['messages'][-1]['content']}")
        except Exception as e:
            print(f"\nAn error occurred: {e}\n")
            break
