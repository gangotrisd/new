import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Initialize Groq LLM
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(model_name="llama-3-3.7b-70b-versatile", api_key=GROQ_API_KEY)

# Load Hugging Face embedding model
hf_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# Function to create FAISS index
def create_faiss_index(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(text)
    embeddings = hf_model.encode(texts)
    embeddings.shape[1]
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, texts
