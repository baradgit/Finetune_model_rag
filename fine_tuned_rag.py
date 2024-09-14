import streamlit as st
import openai
from langchain import OpenAI as LangChainOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain

# Streamlit app title
st.title("RAG and Fine-Tuned Model for Govt Schemes and Contact Info")

# API Key Input
api_key = st.text_input("Enter your OpenAI API Key", type="password")

# Show a warning if the API key is not entered
if not api_key:
    st.warning("Please enter your OpenAI API key to proceed.")
    st.stop()

# Initialize OpenAI client
openai.api_key = api_key  # Set OpenAI API key

# Initialize the LLM to be used by the agent
llm = LangChainOpenAI(openai_api_key=api_key)

# Function to query the fine-tuned model
def query_fine_tuned_model(prompt):
    response = openai.Completion.create(
        model="ft:babbage-002:personal::A7JPv1MB",
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0].text

# Function to extract text from a fixed PDF
def extract_fixed_pdf_text():
    # Fixed PDF path, predefined and not uploaded by the user
    pdf_path = "helpline.pdf"  # Specify the fixed path
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Use LangChain's document loader and text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Extract text from the fixed PDF
with st.spinner("Extracting text from the fixed PDF..."):
    pdf_text = extract_fixed_pdf_text()

# Split the extracted text into documents
documents = text_splitter.split_text(pdf_text)
docs = [Document(page_content=doc) for doc in documents]

# Create an embedding-based retriever using FAISS
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vector_store = FAISS.from_documents(docs, embeddings)

# Function to handle RAG queries from the fixed PDF
def query_pdf_rag(prompt):
    retriever = vector_store.as_retriever()
    
    # Load the QA chain with 'stuff' chain_type
    combine_documents_chain = load_qa_chain(llm, chain_type="stuff")
    
    qa_chain = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=combine_documents_chain,
        return_source_documents=True
    )
    
    # Query the QA chain
    result = qa_chain({"query": prompt})
    return result['result'], result['source_documents']  # Return both result and source documents

# Define tools for the agent
tools = [
    Tool(
        name="Fine-tuned Model",
        func=query_fine_tuned_model,
        description="Useful for answering questions about central govt schemes."
    ),
    Tool(
        name="PDF Retrieval",
        func=query_pdf_rag,
        description="Useful for retrieving contact information from the fixed PDF document."
    )
]

# Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,  # Pass the LLM here
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# User input for querying the agent
query = st.text_input("Ask a question:")

# If a question is entered, run the agent and display the response
if query:
    with st.spinner("Generating response..."):
        response = agent.run(query)
    
    st.write(f"Agent Response: {response}")
