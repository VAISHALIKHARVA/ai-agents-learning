# Job Application Assistant

import re
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.environ["GROQ_API_KEY"])

details = {
    "name": None,
    "email": None,
    "skills": None,
}

def extract_details(text: str):
    name_match = re.search(
        r"(?:my name is|i am|i'm)\s+([A-Za-z]+(?: [A-Za-z]+)*)", text, re.IGNORECASE
    )
    email_match = re.search(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text
    )
    skills_match = re.search(
        r"(?:my skills? (?:are|were?|is|was)|i have experience in|i am skilled in|i know)\s*(.+)",
        text, re.IGNORECASE
    )

    if name_match and not details["name"]:
        details["name"] = name_match.group(1).strip().title()
    if email_match and not details["email"]:
        details["email"] = email_match.group()
    if skills_match and not details["skills"]:
        details["skills"] = skills_match.group(1).strip()

def get_missing() -> list[str]:
    return [k for k, v in details.items() if v is None]

SYSTEM_PROMPT = (
    "You are a job application assistant. Your ONLY goal is to collect the user's "
    "name, email, and skills for their job application.\n\n"
    "STRICT RULES:\n"
    "- NEVER assume, guess, or make up any detail (name, email, or skills).\n"
    "- ONLY use information the user has explicitly provided in this conversation.\n"
    "- Ask for one missing field at a time.\n"
    "- Do NOT mark the application complete until name, email, AND skills are all confirmed."
)

def invoke(history: list) -> AIMessage:
    return llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + history)

if __name__ == "__main__":
    history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        extract_details(user_input)
        history.append(HumanMessage(content=user_input))
        response = invoke(history)
        history.append(AIMessage(content=response.content))
        print(f"Assistant: {response.content}")

        if not get_missing():
            print("All details collected. Application ready!")
            break
