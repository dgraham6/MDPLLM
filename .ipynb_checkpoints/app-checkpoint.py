import os
import streamlit as st
import torch
from langchain.schema import Document
from PyPDF2 import PdfReader
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQAWithSourcesChain
import time
import sys
import sqlite3
import requests

# SQLite3 fix
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Force the use of PyTorch and disable TensorFlow in transformers
os.environ["USE_TF"] = "NO"

# Function to load PDF from URL
def load_pdf_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        with open("temp.pdf", "wb") as f:
            f.write(response.content)
        reader = PdfReader("temp.pdf")
        text = "".join([page.extract_text() for page in reader.pages])
        return text
    except requests.RequestException as e:
        st.error(f"Failed to download PDF from {url}: {str(e)}")
        return ""

@st.cache_resource
def load_data():
    pdf_urls = [
        "https://www.biorxiv.org/content/10.1101/2020.07.28.224253v1.full.pdf",
        "https://www.cartercenter.org/resources/pdfs/health/ephti/library/lecture_notes/health_extension_trainees/generalpathology.pdf"
    ]

    documents_with_metadata = []
    for url in pdf_urls:
        data = load_pdf_from_url(url)
        if data:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
            splits = text_splitter.split_text(data)

            for split in splits:
                documents_with_metadata.append(Document(page_content=split, metadata={"source": url}))

    # Hugging Face Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Check and create Chroma persistence directory if it does not exist
    if not os.path.exists("chromadb"):
         os.makedirs("chromadb")

    # Create Chroma vector store with documents that include metadata
    vectorstore = Chroma.from_documents(
        documents=documents_with_metadata, 
        embedding=embedding_model, 
        persist_directory="chromadb"
    )

    return vectorstore

# Load the data into Chroma Vector Store
vectorstore = load_data()

st.title('PDF-based RAG LLM App with Ollama (LLaMA 3)')

# User input query
query = st.text_input("Enter your query:", "")

# Function to get an answer using Ollama and Chroma
def get_answer(query, max_retries=3, delay=5):
    template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know; don't try to make up an answer.
    Keep the answer as concise as possible, preferably in three sentences.
    {context}
    Question: {question}
    Answer:
    """

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    # Initialize Ollama's LLaMA 3 model using the ollama package directly
    llm = Ollama(model="llama3", temperature=0)

    # Build the QA chain using Ollama and Chroma vector store
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),  # Use the correctly defined vectorstore
        chain_type="stuff"
    )

    # Implement retry logic
    attempt = 0
    while attempt < max_retries:
        try:
            # Run the query and get the answer from the RAG LLM
            result = qa_chain.invoke({
                "question": query,  # Provide the 'question' key here
            })
            return result
        except Exception as e:
            st.warning(f"Connection error or model issue: {str(e)}. Retrying in {delay} seconds...")
            time.sleep(delay)
            attempt += 1
    return None

# Display the answer when the user submits a query
if st.button("Get Answer"):
    if query:
        with st.spinner('Fetching answer...'):
            answer = get_answer(query)
            if answer:
                st.write(answer)
            else:
                st.error("Failed to get a response after multiple attempts.")
    else:
        st.write("Please enter a query to get an answer.")

# Instructions for using Ollama locally
with st.expander("Instructions for Using Ollama"):
    st.markdown("""
    ### How to Use Ollama with This App
    1. **Install Ollama**: 
       - macOS: `brew install ollama`
       - Linux: Run the following:
         ```
         curl -fsSL https://ollama.com/install.sh | sh
         ```
    2. **Run the Ollama server** (Optional for Docker):
       - Ollama can run models locally without needing `ollama serve`:
         ```
         ollama pull llama3
         ```
       - You no longer need to connect via an HTTP server, the Python API will handle the local LLM calls directly.
    """)