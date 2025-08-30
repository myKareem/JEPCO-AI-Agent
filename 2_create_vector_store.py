import os
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma # <-- Import Chroma
from langchain_community.document_loaders import TextLoader

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Configuration ---
DATA_PATH = 'data/combined_text.txt'
# Use a different path for the new DB
DB_CHROMA_PATH = 'vectorstore/db_chroma' 

# ==============================================================================
# The Data Ingestion part (scraping and PDF extraction) is exactly the same.
# We include it here to make the script self-contained.
# ==============================================================================

def scrape_urls(urls):
    """Scrapes the text content from a list of URLs."""
    all_text = ""
    for url in urls:
        try:
            response = requests.get(url, verify=False)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            all_text += soup.get_text(separator='\n', strip=True) + "\n\n"
        except requests.exceptions.RequestException as e:
            print(f"Error scraping {url}: {e}")
    return all_text

def extract_pdf_text(pdf_path):
    """Extracts text from a PDF file."""
    if not os.path.exists(pdf_path):
        print(f"Warning: PDF file not found at {pdf_path}")
        return ""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def create_vector_store():
    """Ingests data and creates a persistent Chroma vector store."""
    
    # --- 1. Ingest Data ---
    urls_to_scrape = [
        "https://www.jepco.com.jo/ar/Home/%d8%b3%d8%af%d8%af-%d9%81%d8%a7%d8%aa%d9%88%d8%b1%d8%aa%d9%83",
        "https://www.jepco.com.jo/ar/Home/%d8%af%d9%88%d8%a7%d8%a6%d8%b1-%d8%a7%d9%84%d8%a7%d8%b4%d8%aa%d8%b1%d8%a7%d9%83%d8%a7%d8%aa",
        "https://www.jepco.com.jo/ar/Home/%d8%a7%d8%aa%d8%b5%d9%84-%d8%a8%d9%86%d8%a7",
        "https://www.jepco.com.jo/ar/Home/%d8%a7%d9%84%d8%b7%d8%a7%d9%82%d8%a9-%d8%a7%d9%84%d9%85%d8%aa%d8%ac%d8%af%d8%af%d8%a9",
        "https://www.jepco.com.jo/ar/Home/%D8%A7%D9%84%D8%A7%D8%B3%D8%A6%D9%84%D8%A9-%D8%A7%D9%84%D8%B4%D8%A7%D8%A6%D8%B9%D8%A9"
    ]
    print("Scraping websites...")
    scraped_text = scrape_urls(urls_to_scrape)

    pdf_file_path = 'data/doc.pdf'
    print("Extracting text from PDF...")
    pdf_text = extract_pdf_text(pdf_file_path)

    # Save to a temporary text file to be loaded
    if not os.path.exists('data'):
        os.makedirs('data')
    with open(DATA_PATH, 'w', encoding='utf-8') as f:
        f.write(scraped_text + "\n\n" + pdf_text)
    print(f"Data ingestion complete. Combined text saved to '{DATA_PATH}'")

    # --- 2. Load and Split Documents ---
    loader = TextLoader(DATA_PATH, encoding='utf-8')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # --- 3. Define Embeddings ---
    # Using the same multilingual model for consistency
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    # --- 4. Create and Persist the Chroma DB ---
    print(f"Creating and persisting ChromaDB at '{DB_CHROMA_PATH}'...")
    db = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_CHROMA_PATH
    )
    print("✅ Vector store created and saved successfully.")

if __name__ == "__main__":
    create_vector_store()
