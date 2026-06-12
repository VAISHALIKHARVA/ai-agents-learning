# Job Application Assistant

import re
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=os.environ["GROQ_API_KEY"])

memory = MemorySaver()

# store the user details in memory
details = {
    "name": None,
    "email": None,
    "skills": None,
}

def extract_details(text):
    name_match = re.search(r"(?:my name is | i am)\s+([A-Z][a-z]+(?: [A-Z][a-z]+)*)", text, re.IGNORECASE)
    email_match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
    skills_match = re.search(r"(?:my skills are | i have experience in | i am skilled in )(.+)", text, re.IGNORECASE)

    response = []

    if name_match:
        details["name"] = name_match.group(1).title()
        response.append(f"Name: {name_match.group(1)} saved.")

    if email_match:
        details["email"] = email_match.group()
        response.append(f"Email: {email_match.group()} saved.")

    if skills_match:
        details["skills"] = skills_match.group(1)
        response.append(f"Skills: {skills_match.group(1)} saved.")

    if not any([name_match, email_match, skills_match]):
        return "No details found. Please provide your name, email, and skills."
    
    return "\n".join(response) + "Let me check what else i need."

def check_details(_text: str = ""):
    missing = [key for key, value in details.items() if value is None]
    if missing:
        return f"Please provide the following details: {', '.join(missing)}."
    return "All details are complete. I can now help you with your job application."

SYSTEM_MESSAGE = (
    "You are a job application assistant. Your ONLY goal is to collect the user's "
    "name, email, and skills for their job application.\n\n"
    "STRICT RULES:\n"
    "- NEVER assume, guess, or make up any detail (name, email, or skills).\n"
    "- ONLY use information the user has explicitly typed in this conversation.\n"
    "- If any detail is missing, ask the user to provide it directly.\n"
    "- Always call 'check_user_details' before declaring all details complete.\n"
    "- Do NOT mark the application complete until name, email, AND skills are all confirmed."
)

@tool
def extract_user_details(text: str) -> str:
    """Extracts user details such as name, email, and skills from the input text."""
    return extract_details(text)

@tool
def check_user_details(_text: str = "") -> str:  # type: ignore[unused-variable]
    """Checks if all user details are provided."""
    return check_details()

agent = create_agent(
    model=llm,
    tools=[extract_user_details, check_user_details],
    system_prompt=SYSTEM_MESSAGE,
    checkpointer=memory,
)

# Iteraction flow
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    config = {"configurable": {"thread_id": "job-application-session"}}
    response = agent.invoke({"messages": [{"role": "user", "content": user_input}]}, config=config)
    print(f"Assistant: {response['messages'][-1].content}")

    # Goal achieved when all details are collected
    if all(value is not None for value in details.values()):
        print("Great! I have all the information I need to help you with your job application.")
        break