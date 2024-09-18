import streamlit as st
import pdfplumber
import openai
from langchain import OpenAI as LangChainOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
import logging

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
        return f"Error extracting text from PDF: {str(e)}"

# Create the prompt based on the extracted PDF text and the user's query
def create_prompt(text, query):
    return f"Using the following extracted text from a government document: \"{text}\", please answer the question: {query}"

# Get response from the fine-tuned model or fall back to other solutions
def get_fine_tuned_model_response(prompt, api_key):
    openai.api_key = api_key
    try:
        # Attempt to use the fine-tuned model first
        response = openai.Completion.create(
            model="ft:gpt-4o-mini-2024-07-18:personal::A7RpYRkF",  # Replace with the fine-tuned model ID
            prompt=prompt,
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        logging.error(f"Error with fine-tuned model: {str(e)}")
        return None

# Get response using the PDF extracted data
def get_pdf_based_response(query, api_key):
    pdf_text = extract_fixed_pdf_text()
    if "Error" in pdf_text:
        return pdf_text  # Return the error if something went wrong with the PDF extraction
    prompt = create_prompt(pdf_text[:2000], query)  # Limit to first 2000 characters
    return get_fine_tuned_model_response(prompt, api_key)

# Fall back to GPT-4 or any other model silently without notifying the user about the fallback
def fallback_model_response(query, api_key):
    openai.api_key = api_key
    try:
        # Use GPT-4 (or another powerful model) as a fallback
        response = openai.Completion.create(
            model="gpt-4",  # Using GPT-4 silently as a fallback
            prompt=query,
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        logging.error(f"Error with fallback model: {str(e)}")
        return "Unable to provide an answer at this time."

# Function to handle RAG queries using LangChain agents with fallback logic
def query_rag_agent(api_key, query):
    # Set up the LangChain agent tools
    tools = [
        Tool(
            name="Fine-Tuned Model",
            func=lambda q: get_fine_tuned_model_response(q, api_key) or fallback_model_response(q, api_key),
            description="This tool uses the fine-tuned model to answer queries.",
        ),
        Tool(
            name="PDF-Based Response",
            func=lambda q: get_pdf_based_response(q, api_key) or fallback_model_response(q, api_key),
            description="This tool extracts information from a government document PDF to answer queries."
        )
    ]

    # Initialize the agent with the tools
    llm = LangChainOpenAI(openai_api_key=api_key)  # Initialization, won't be used directly
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
        logging.error(f"Error in agent execution: {str(e)}")
        return fallback_model_response(query, api_key)

# Streamlit application
st.title("Government Schemes Information")

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
        st.write("Answer:")
        st.write(response)
