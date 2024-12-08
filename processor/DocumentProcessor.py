import re
import unicodedata
import docx
import PyPDF2
import os

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
    def extract_text(file_path: str) -> str:
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