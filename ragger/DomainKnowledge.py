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
from transformers import pipeline


class DomainKnowledgeRAG:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        #embedding_model_name: str = "mixedbread-ai/mxbai-embed-large-v1",
        llm_model_name: str = "microsoft/Orca-2-13b",
        #llm_model_name: str = "HuggingFaceH4/zephyr-7b-beta"
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize RAG system with embedding and language models

        Args:
            embedding_model_name (str): Embedding model for semantic search
            llm_model_name (str): Language model for generation
            device (str): Computation device
        """
        self.device = device
        
        hfapi_key = getpass("Enter you HuggingFace access token:")
        os.environ["HF_TOKEN"] = hfapi_key
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hfapi_key
        
        encode_kwargs = {'normalize_embeddings': False}
        
        # Setup LLM
        #self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        #self.model = AutoModelForCausalLM.from_pretrained(
        #    llm_model_name,
        #    device_map=self.device
            #torch_dtype=torch.float16
        #)

        # Create Text Generation Pipeline
        self.generator = pipeline(
            "text-generation",
            model=llm_model_name,
            #tokenizer=self.tokenizer,
            #max_length=2048
        )
        self.llm = HuggingFacePipeline(pipeline=self.generator)

        # Vector Store
        self.vector_store = None
        
        # Setup Embeddings
        self.embeddings = HuggingFaceEmbeddings(embedding_model_name)


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


    def generate_synthetic_data(self, existing_docs, num_synthetics: int = 5):
        """
        Generate synthetic training data

        Args:
            existing_docs (List): Existing documents
            num_synthetics (int): Number of synthetic documents to generate

        Returns:
            List of synthetic documents
        """
        synthetic_docs = []

        prompt_template = """
        Based on the following SAP/SFDC domain context, generate a synthetic
        business requirement document excerpt:

        Context: {context}

        Synthetic Document Excerpt:
        """

        for doc in existing_docs[:num_synthetics]:
            synthetic_excerpt = self.llm(
                prompt_template.format(context=doc.page_content)
            )
            synthetic_docs.append(synthetic_excerpt)

        return synthetic_docs