from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.environ["GROQ_API_KEY"])
# llm = ChatGroq(
#     api_key="YOUR_API_KEY",
#     model="llama-3.3-70b-versatile"
# )

class State(TypedDict):
    question: str
    answer: str

def chatbot(state: State):
    response = llm.invoke(state["question"])

    return {
        "answer": response.content
    }

graph = StateGraph(State)

graph.add_node("chatbot", chatbot)

graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

app = graph.compile()

result = app.invoke({
    "question": "What is LangGraph?"
})

print(result)