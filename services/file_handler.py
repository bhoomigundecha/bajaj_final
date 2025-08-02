import requests
import pdfplumber
import docx
import tempfile
import os
from fastapi import HTTPException

def fetch_and_extract_text(file_url: str) -> str:
    try:
        response = requests.get(file_url)
        if response.status_code != 200:
            raise Exception("download issue")

        content_type = response.headers.get("content-type", "").lower()
        _, temp_path = tempfile.mkstemp()
        os.close(_)
        with open(temp_path, "wb") as f:
            f.write(response.content)

        file_url_str = str(file_url).lower()

        if ".pdf" in file_url_str or "pdf" in content_type:
            text = extract_pdf(temp_path)
        elif ".docx" in file_url_str or "word" in content_type:
            text = extract_docx(temp_path)
        elif ".txt" in file_url_str or "text" in content_type:
            text = extract_txt(temp_path)
        else:
            raise Exception("file type match issue")

        os.remove(temp_path)
        return text

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"file processing issue: {e}")

def extract_pdf(path: str) -> str:
    try:
        # if it creates problem use the commented code
        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
        # print(f"Opening PDF: {path}")
        # with pdfplumber.open(path) as pdf:
        #     # print(f"PDF has {len(pdf.pages)} pages")
        #     texts = []
        #     for i, page in enumerate(pdf.pages):
        #         try:
        #             # print(f"Extracting page {i+1}...")
        #             text = page.extract_text()
        #             # print(f"Page {i+1} text length: {len(text) if text else 0}")
        #             if text and text.strip():
        #                 texts.append(text.strip())
        #                 # print(f"Page {i+1} text added")
        #             else:
        #                 print(f"Page {i+1} has no extractable text")
        #         except Exception as page_error:
        #             print(f"Error extracting page {i+1}: {page_error}")
        #             continue
            
        #     # print(f"Total pages with text: {len(texts)}")
        #     if not texts:
        #         raise Exception("No text could be extracted from any page")
            
        #     # print("Joining texts...")
        #     result = "\n\n".join(texts)
        #     # print(f"PDF extraction complete. Total length: {len(result)}")
        #     return result
    except Exception as e:
        print(f"PDF extraction failed: {e}")
        raise Exception(f"pdf: {e}")

def extract_docx(path: str) -> str:
    try:
        doc = docx.Document(path)
        return "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        raise Exception(f"doc: {e}")

def extract_txt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise Exception(f"txt: {e}")
