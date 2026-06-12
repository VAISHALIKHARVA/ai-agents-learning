import streamlit as st
import importlib.util
import uuid
import os
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(
    page_title="Job Application Assistant",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def load_agent():
    spec = importlib.util.spec_from_file_location(
        "goal_based_agents",
        os.path.join(os.path.dirname(__file__), "goal-based-agents.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mod = load_agent()

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "goal_achieved" not in st.session_state:
    st.session_state.goal_achieved = False
if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None

def reset_chat():
    st.session_state.messages = []
    st.session_state.history = []
    st.session_state.goal_achieved = False
    st.session_state.last_uploaded = None
    for key in mod.details:
        mod.details[key] = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 Upload Resume (Optional)")
    st.caption("Upload your resume")

    uploaded_file = st.file_uploader(
        label="upload",
        type=["pdf", "txt"],
        label_visibility="collapsed"
    )

    if uploaded_file and uploaded_file.name != st.session_state.last_uploaded:
        st.session_state.last_uploaded = uploaded_file.name
        resume_text = ""

        if uploaded_file.type == "text/plain":
            resume_text = uploaded_file.read().decode("utf-8")
        else:
            try:
                import pypdf
                reader = pypdf.PdfReader(uploaded_file)
                resume_text = "\n".join(
                    page.extract_text() or "" for page in reader.pages
                )
            except ImportError:
                st.warning("Install pypdf for PDF support: `pip install pypdf`")

        if resume_text.strip():
            mod.extract_details(resume_text)
            st.session_state.history.append(HumanMessage(content=f"Here is my resume:\n{resume_text}"))
            response = mod.invoke(st.session_state.history)
            st.session_state.history.append(AIMessage(content=response.content))
            st.session_state.messages.append({"role": "user", "content": f"📄 Resume uploaded: {uploaded_file.name}"})
            st.session_state.messages.append({"role": "assistant", "content": response.content})
            st.rerun()

    st.button("🔄 Reset Chat", on_click=reset_chat, use_container_width=True)

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🧠 Goal-Based Agent: Job Application Assistant")
st.markdown("Tell me your **name**, **email**, and **skills** to complete your application!")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if st.session_state.goal_achieved:
    st.success("🎉 All details collected! Your job application is ready.")

if not st.session_state.goal_achieved:
    user_input = st.chat_input("Type here...")
    if user_input:
        mod.extract_details(user_input)
        st.session_state.history.append(HumanMessage(content=user_input))

        response = mod.invoke(st.session_state.history)

        st.session_state.history.append(AIMessage(content=response.content))
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": response.content})

        if not mod.get_missing():
            st.session_state.goal_achieved = True

        st.rerun()
