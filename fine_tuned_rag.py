import streamlit as st
import pdfplumber
import openai
from langchain import OpenAI as LangChainOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
import requests  # For web search functionality
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
            model="gpt-4o-mini-2024-07-18:personal::A7RpYRkF",  # Fine-tuned model identifier
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant specializing in government schemes and policies."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,  # Adjust the token limit based on expected answer length
            temperature=0.9,
        )
        model_response = response['choices'][0]['message']['content'].strip()
        # Check if the model response is generic or unhelpful, triggering a web search
        if "I am not able" in model_response or "Unfortunately" in model_response:
            return None  # Return None to trigger web search fallback
        return model_response
    except Exception as e:
        return None  # If there's an issue with the model, trigger web search

# Fallback web search using a public search engine API
def web_search(query):
    search_api_key = "YOUR_WEB_SEARCH_API_KEY"  # Use a valid web search API key (e.g., Bing)
    search_url = f"https://api.bing.microsoft.com/v7.0/search?q={query}"
    headers = {"Ocp-Apim-Subscription-Key": search_api_key}

    try:
        response = requests.get(search_url, headers=headers)
        if response.status_code == 200:
            search_results = response.json()
            # Process and extract relevant information from search results
            if 'webPages' in search_results:
                top_results = search_results['webPages']['value'][:3]  # Limit to top 3 results
                return "\n\n".join([result['snippet'] for result in top_results])
            else:
                return "No relevant information found through web search."
        else:
            return "Error in web search."
    except Exception as e:
        return f"Error in web search: {str(e)}"

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
            func=lambda q: get_fine_tuned_model_response(prompt, api_key) or web_search(q),  # Try fine-tuned, then fallback to web search
            description="This tool uses the fine-tuned model to answer queries. Falls back to web search if no suitable answer."
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
st.title("RAG with Fine-Tuned Model and Web Search Fallback")

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
