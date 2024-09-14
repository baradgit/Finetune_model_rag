import streamlit as st
import pdfplumber
import openai
from langchain import OpenAI as LangChainOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Extract text from a predefined PDF file
def extract_fixed_pdf_text():
    pdf_path = "helpline.pdf"  # Fixed path to the PDF file
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error: {str(e)}"

# Create the prompt based on the extracted PDF text and the user's query
def create_prompt(text, query):
    return f"Using the following extracted text from a government document: \"{text}\", please answer the question: {query}"

# Get response from the fine-tuned OpenAI model with temperature control
def get_fine_tuned_model_response(prompt, api_key):
    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model="ft:babbage-002:personal::A7JPv1MB",  # Fine-tuned model identifier
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant specializing in government schemes and policies."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,  # Adjust the token limit based on expected answer length
            temperature=0.1  # Set temperature to 0.1 for deterministic output
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Function to handle RAG queries using LangChain agents with fixed temperature
def query_rag_agent(api_key, query):
    # Extract text from the fixed PDF
    pdf_text = extract_fixed_pdf_text()

    if "Error" in pdf_text:
        return pdf_text

    # Create the prompt based on the PDF text
    prompt = create_prompt(pdf_text[:2000], query)  # Limiting text to the first 2000 characters for prompt generation

    # Set up the LangChain agent tools
    tools = [
        Tool(
            name="Fine-tuned Model",
            func=lambda q: get_fine_tuned_model_response(prompt, api_key),  # Fine-tuned model tool
            description="This tool uses the fine-tuned model to answer queries."
        )
    ]

    # Initialize the agent with the tool
    llm = LangChainOpenAI(openai_api_key=api_key)  # This is only for initialization, won't be used directly
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # Run the agent with the user's query
    try:
        return agent.run(query)
    except Exception as e:
        return f"Error in agent execution: {str(e)}"

# Streamlit application
st.title("RAG with Fine-Tuned Model and Agent")

# API key input
api_key = st.text_input("Enter your OpenAI API key:", type="password")

# Check if API key is provided
if not api_key:
    st.write("Please enter your OpenAI API key to continue.")
    st.stop()

# User query input
query = st.text_input("Enter your query:")

# If the user provides a query, run the agent
if query:
    with st.spinner("Generating response..."):
        response = query_rag_agent(api_key, query)
    
    st.write("Response from the Fine-Tuned Model:")
    st.write(response)
