from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_chroma import Chroma
import torch
import re, unicodedata

def load_documents(document_paths: List[str]) -> List[Dict]:
    documents = []

    for path in document_paths:
        if path.lower().endswith('.pdf'):
            loader = PyPDFLoader(path)
        elif path.lower().endswith(('.docx', '.doc')):
            loader = Docx2txtLoader(path)
        else:
            print(f"Unsupported file type: {path}")
            continue

        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        documents.extend(text_splitter.split_documents(docs))

    return documents

def splitDoc(documents):

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap = 128
    )

    splits = text_splitter.split_documents(documents)
    return splits

def getEmbedding():
    modelPath="mixedbread-ai/mxbai-embed-large-v1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {'device': device}      # cuda/cpu
    encode_kwargs = {'normalize_embeddings': False}

    embedding =  HuggingFaceEmbeddings(
        model_name=modelPath,     
        model_kwargs=model_kwargs, 
        encode_kwargs=encode_kwargs 
    )
    return embedding

def clean_text(text: str) -> str:
    """Clean and normalize input text."""
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^.*?(\d+)$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

if __name__ == "__main__":

    assessment_document_paths = [
    'new_assessment - Copy.pdf'
    ]
    
    documents = load_documents(assessment_document_paths)
    splits = splitDoc(documents)
    embeddings = getEmbedding()
    persist_directory = 'docs/chroma/'

    vectordb = Chroma.from_documents(
        documents=splits, # splits we created earlier
        embedding=embeddings,
        persist_directory=persist_directory # save the directory
        )

    question_1 = "What are the objectives of Interface Assessment?"
    docs = vectordb.search(question_1, search_type="mmr", k=5)
    print(f"Responding to the {question_1}:")
    
    response = ""
    for i in range(len(docs)):
        response = response + docs[i].page_content

    print(clean_text(response))

    #Similarly add as many questions as needed
    #And include those response in few shots prompt template