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
import requests

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
            model="gpt-4o-mini-2024-07-18:personal::A7RpYRkF",  # Fine-tuned model identifier
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant specializing in government schemes and policies."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,  # Adjust the token limit based on expected answer length
            temperature=0.9,  # Set temperature to control randomness
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Fallback to web search using SerpAPI (or other API) when fine-tuned model fails
def web_search(query):
    # Here, I'm using SerpAPI as an example; replace with your web search API of choice
    api_key = "your_serpapi_key"
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key
    }
    try:
        response = requests.get("https://serpapi.com/search", params=params)
        data = response.json()
        if "organic_results" in data:
            # Return the first result snippet
            return data["organic_results"][0]["snippet"]
        else:
            return "No relevant web search results found."
    except Exception as e:
        return f"Error: Web search failed: {str(e)}"

# Fallback to GPT-3.5/4 when web search and fine-tuned model fail
def get_gpt_response(prompt, api_key):
    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Fallback to GPT-3.5 or GPT-4
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant specializing in government schemes and policies."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,  # Adjust the token limit
            temperature=0.7,  # Adjust the temperature as needed
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Function to handle RAG queries using LangChain agents with fallback mechanisms
def query_rag_agent(api_key, query):
    # Extract text from the fixed PDF
    pdf_text = extract_fixed_pdf_text()

    if "Error" in pdf_text:
        return pdf_text

    # Create the prompt based on the PDF text
    prompt = create_prompt(pdf_text[:2000], query)  # Limiting text to the first 2000 characters for prompt generation

    # Try the fine-tuned model first
    fine_tuned_response = get_fine_tuned_model_response(prompt, api_key)
    if "Error" in fine_tuned_response:
        # If fine-tuned model fails, try web search
        st.warning("Fine-tuned model failed, attempting web search...")
        web_response = web_search(query)
        if web_response != "No relevant web search results found.":
            return web_response
        else:
            # Fallback to GPT-3.5/4 if web search fails
            st.warning("Web search failed, falling back to GPT-3.5/4...")
            return get_gpt_response(prompt, api_key)
    else:
        return fine_tuned_response

# Streamlit application
st.title("RAG with Fine-Tuned Model, Web Search, and GPT Fallback")

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
    
    st.write("Response:")
    st.write(response)
