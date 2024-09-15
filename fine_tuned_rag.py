import streamlit as st
import pdfplumber
import openai

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

# Get response directly from the OpenAI model
def get_model_response(prompt, api_key):
    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Using GPT-3.5 directly
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant specializing in government schemes and policies."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,  # Adjust the token limit based on expected answer length
            temperature=0.7,  # Set temperature for balanced response
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {str(e)}"

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

# If the user provides a query, run the model directly
if query:
    with st.spinner("Generating response..."):
        # Extract PDF text and prepare prompt
        pdf_text = extract_fixed_pdf_text()

        if "Error" in pdf_text:
            st.write(pdf_text)
        else:
            prompt = create_prompt(pdf_text[:2000], query)  # Limit text for prompt

            # Call OpenAI API directly
            response = get_model_response(prompt, api_key)
        
            st.write("Answer:")
            st.write(response)
