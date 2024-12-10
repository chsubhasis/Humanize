import re
import unicodedata
import docx
import PyPDF2
import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

class SAPDocumentProcessor:

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize input text."""
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        text = unicodedata.normalize('NFKD', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'^.*?(\d+)$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def extract_text_simple(file_path: str) -> str:
        """Extract text from PDF or DOCX files."""
        _, ext = os.path.splitext(file_path)
        
        try:
            if ext.lower() == '.pdf':
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ' '.join([page.extract_text() for page in reader.pages])
            elif ext.lower() in ['.docx', '.doc']:
                doc = docx.Document(file_path)
                text = '\n'.join([para.text for para in doc.paragraphs])
            else:
                raise ValueError(f"Unsupported file type: {ext}")
            
            return SAPDocumentProcessor.clean_text(text)
        
        except Exception as e:
            raise RuntimeError(f"Error extracting text from {file_path}: {str(e)}")
    
    @staticmethod
    def extract_text_with_llm(file_path: str) -> str:

        mydoc = []

        if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.lower().endswith(('.docx', '.doc')):
            loader = Docx2txtLoader(file_path)
        else:
            print(f"Unsupported file type: {file_path}")

        docs = loader.load()
        #print("docs", docs)

        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
        
        split_docs = text_splitter.split_documents(docs)
        page_content = [doc.page_content for doc in split_docs]
        page_content = clean_text("\n".join(page_content))

        extraction_prompt = """
        You are a specialist in reading SAP assessment reports in PDF or Docs. 
        Extract the following key information from the assessment document.
            1. Assessment Summary.
            2. Observations and Suggestions for improvement.
            3. Existing key issues and factors. Root cause of those issues and gaps.
            4. Roadmap

        Document Context: {assessment_document}

        Extracted Information:
        """

        llm = HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            task="summarization",
            )

        extracted_info = llm.invoke(
            extraction_prompt.format(assessment_document=page_content),
            top_k = 30,
            temperature = 0.1,
            repetition_penalty = 1.03,
        )

        with open("assessment_summary.txt", 'w') as f:
            f.write(extracted_info)

        return extracted_info