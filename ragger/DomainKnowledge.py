import logging
from typing import List, Dict, Any
import os
from getpass import getpass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEndpoint
from transformers import pipeline

class DomainKnowledgeRAG:
    def __init__(
        self,
        embedding_model_name: str = "mixedbread-ai/mxbai-embed-large-v1",
        llm_model_name: str = "HuggingFaceH4/zephyr-7b-beta",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):

        hfapi_key = getpass("Enter you HuggingFace access token:")
        os.environ["HF_TOKEN"] = hfapi_key
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hfapi_key
        
        self.llm = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            task="text-generation",
            max_new_tokens = 512,
            top_k = 30,
            temperature = 0.1,
            repetition_penalty = 1.03,
        )

        # Vector Store
        self.vector_store = None
        
        model_kwargs = {'device': device}      # cuda/cpu
        encode_kwargs = {'normalize_embeddings': False}

        self.embeddings =  HuggingFaceEmbeddings(
            model_name=embedding_model_name,     
            model_kwargs=model_kwargs, 
            encode_kwargs=encode_kwargs
        )

    def load_documents(self, document_paths: List[str]) -> List[Dict]:
        """
        Load and preprocess documents from various sources

        Args:
            document_paths (List[str]): Paths to assessment documents

        Returns:
            List of preprocessed document chunks
        """
        documents = []

        for path in document_paths:
            if path.lower().endswith('.pdf'):
                loader = PyPDFLoader(path)
            elif path.lower().endswith(('.docx', '.doc')):
                loader = Docx2txtLoader(path)
            else:
                logging.warning(f"Unsupported file type: {path}")
                continue

            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )

            documents.extend(text_splitter.split_documents(docs))

        return documents

    def create_vector_store(self, documents):
        """
        Create vector store for semantic search

        Args:
            documents (List): Preprocessed document chunks
        """
        self.vector_store = FAISS.from_documents(documents, self.embeddings)