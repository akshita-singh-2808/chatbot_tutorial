import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

st.title("🤖 Gemini Chatbot")

model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=os.getenv("Google_API_KEY"),
    temperature=0.7
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    else:
        with st.chat_message("assistant"):
            st.write(message.content)

# Input
user_input = st.chat_input("Type your message...")

if user_input:

    st.session_state.chat_history.append(
        HumanMessage(content=user_input)
    )

    with st.chat_message("user"):
        st.write(user_input)

    result = model.invoke(st.session_state.chat_history)

    # Extract only the actual text
    response = result.content[0]["text"]

    st.session_state.chat_history.append(
        AIMessage(content=response)
    )

    with st.chat_message("assistant"):
        st.write(response)