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
