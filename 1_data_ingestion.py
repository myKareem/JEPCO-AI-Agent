import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import os

# --- Web Scraping ---
def scrape_urls(urls):
    """Scrapes the text content from a list of URLs."""
    all_text = ""
    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes
            soup = BeautifulSoup(response.content, 'html.parser')
            # Extract text from the body, you might need to adjust this
            # based on the website's structure
            all_text += soup.get_text(separator='\n', strip=True) + "\n\n"
        except requests.exceptions.RequestException as e:
            print(f"Error scraping {url}: {e}")
    return all_text

# --- PDF Extraction ---
def extract_pdf_text(pdf_path):
    """Extracts text from a PDF file."""
    if not os.path.exists(pdf_path):
        return f"Error: PDF file not found at {pdf_path}"
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# --- Main Data Ingestion ---
if __name__ == "__main__":
    # 1. List of URLs to scrape
    urls_to_scrape = [
        "https://www.jepco.com.jo/ar/Home/%d8%b3%d8%af%d8%af-%d9%81%d8%a7%d8%aa%d9%88%d8%b1%d8%aa%d9%83",
        "https://www.jepco.com.jo/ar/Home/%d8%af%d9%88%d8%a7%d8%a6%d8%b1-%d8%a7%d9%84%d8%a7%d8%b4%d8%aa%d8%b1%d8%a7%d9%83%d8%a7%d8%aa",
        "https://www.jepco.com.jo/ar/Home/%d8%a7%d8%aa%d8%b5%d9%84-%d8%a8%d9%86%d8%a7",
        "https://www.jepco.com.jo/ar/Home/%d8%a7%d9%84%d8%b7%d8%a7%d9%82%d8%a9-%d8%a7%d9%84%d9%85%d8%aa%d8%ac%d8%af%d8%af%d8%a9",
        "https://www.jepco.com.jo/ar/Home/%D8%A7%D9%84%D8%A7%D8%B3%D8%A6%D9%84%D8%A9-%D8%A7%D9%84%D8%B4%D8%A7%D8%A6%D8%B9%D8%A9"
        
    ]
    print("Scraping websites...")
    scraped_text = scrape_urls(urls_to_scrape)

    # 2. Path to your PDF file
    pdf_file_path = 'data/doc.pdf'
    print("Extracting text from PDF...")
    pdf_text = extract_pdf_text(pdf_file_path)

    # 3. Combine all text
    combined_text = scraped_text + "\n\n" + pdf_text

    # 4. Save the combined text to a file
    with open('data/combined_text.txt', 'w', encoding='utf-8') as f:
        f.write(combined_text)

    print("Data ingestion complete. Combined text saved to 'data/combined_text.txt'")