import streamlit as st
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from local_agent import retriever


model = OllamaLLM(model="gemma:2b")


template = """
You are an expert in answering questions about a pizza restaurant.

Here are some relevant reviews:
{reviews}

Here is the question to answer:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


st.set_page_config(page_title="🍕 Pizza Review AI Chat", page_icon="🍕")
st.title("🍕 Restaurant Chatbot")
st.markdown("Ask anything about the restaurant based on customer reviews.")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


user_question = st.chat_input("Ask your question about the restaurant...")

if user_question:
    with st.spinner("Thinking..."):
        reviews = retriever.invoke(user_question)
        response = chain.invoke({"reviews": reviews, "question": user_question})
    
    
    st.session_state.chat_history.append(("user", user_question))
    st.session_state.chat_history.append(("ai", response))


for role, message in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").write(message)
    else:
        st.chat_message("assistant").write(message)
