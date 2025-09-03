import requests
from bs4 import BeautifulSoup
import PyPDF2
import io
import json
import os

def scrape_webpage(url):
    """Scrapes clean text from a web page URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15,verify=False)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # This selector is chosen to target the main content area on jepco.com.jo
        main_content = soup.find('div', class_='sf_colsOut content_main') or soup.body
        return main_content.get_text(separator='\n', strip=True) if main_content else ""
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return None

def scrape_local_pdf(filepath):
    """Extracts text from a local PDF file."""
    try:
        with open(filepath, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
    except FileNotFoundError:
        print(f"Error: PDF file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading PDF {filepath}: {e}")
        return None

def main():
    """Main function to scrape all sources and save raw content."""
    
    # --- CONFIGURE YOUR SOURCES HERE ---
    sources_to_process = {
        "webpages": [
            # Feel free to change the category to be more specific.
            {"url": "https://www.jepco.com.jo/ar/home", "category": "General"},
            {"url": "https://www.jepco.com.jo/ar/Home/اتصل-بنا", "category": "Contact"},
            {"url": "https://www.jepco.com.jo/ar/Home/الكلمة-الترحيبية", "category": "About"},
            {"url": "https://www.jepco.com.jo/ar/Home/الرؤيا-الرسالة", "category": "About"},
            {"url": "https://www.jepco.com.jo/ar/Home/تاريخ-الشركة", "category": "About"},
            {"url": "https://www.jepco.com.jo/ar/Home/النظرة-المستقبلية-للشركة", "category": "About"},
            {"url": "https://www.jepco.com.jo/ar/Home/كلمة-رئيس-مجلس-الإدارة", "category": "About"},
            {"url": "https://www.jepco.com.jo/ar/Home/مجلس-الادارة", "category": "About"},
            {"url": "https://www.jepco.com.jo/ar/Home/الإدارة-العليا-التنفيذية", "category": "About"},
            {"url": "https://www.jepco.com.jo/ar/Home/الهيكل-التنظيمي", "category": "About"},
            {"url": "https://www.jepco.com.jo/ar/Home/أهم-المشاريع", "category": "Projects"},
            {"url": "https://www.jepco.com.jo/ar/Home/ServiceStepPage", "category": "Services"},
            {"url": "https://www.jepco.com.jo/ar/Home/سدد-فاتورتك", "category": "Billing"},
            {"url": "https://www.jepco.com.jo/ar/Home/دوائر-الاشتراكات", "category": "Services"},
            {"url": "https://www.jepco.com.jo/ar/Home/الطاقة-المتجددة", "category": "Services"},
            {"url": "https://www.jepco.com.jo/ar/Home/التسديد-الإلكتروني-اي-فواتيركم", "category": "Billing"},
            {"url": "https://www.jepco.com.jo/ar/Home/لأنا_معك", "category": "Initiatives"},
            {"url": "https://www.jepco.com.jo/ar/Home/مشروع-العدادات-و-الشبكات-الذكية", "category": "Projects"},
            {"url": "https://www.jepco.com.jo/ar/Home/شهادات-التميز", "category": "About"},
            {"url": "https://www.jepco.com.jo/ar/Home/مشروع-الحوسبة-و-التحول-الرقمي-و-أمن-المعلومات", "category": "Projects"},
            {"url": "https://www.jepco.com.jo/ar/Home/الهوية-البصرية", "category": "About"},
            {"url": "https://www.jepco.com.jo/ar/Home/مشروع-التواصل-و-التوعية", "category": "Projects"},
            {"url": "https://www.jepco.com.jo/ar/Home/التوعية_بالأمن_السيبراني", "category": "General"},
            {"url": "https://www.jepco.com.jo/ar/Home/عرض-المناقصات-الحالية", "category": "Tenders"},
            {"url": "https://www.jepco.com.jo/ar/Home/الأرباح_خلال_آخر_5_%20سنوات", "category": "Investor"},
            {"url": "https://www.jepco.com.jo/ar/Home/عروض_توضيحية", "category": "Investor"},
            {"url": "https://www.jepco.com.jo/ar/Home/التقارير_السنوية", "category": "Investor"},
            {"url": "https://www.jepco.com.jo/ar/Home/وسائل-التواصل", "category": "Contact"}
        ],
        "local_pdfs": [
            # IMPORTANT: Replace these paths with the actual paths to your PDF files.
            # Use double backslashes (\\) in the path on Windows.
            {"path": "C:\\Users\\20220458\\Desktop\\Dataset\\الحوكمة.pdf", "category": "Tariffs"},
            {"path": "C:\\Users\\20220458\\Desktop\\Dataset\\تقرير الاستدامة 2024.pdf", "category": "Investor"},
            {"path": "C:\\Users\\20220458\\Desktop\\Dataset\\دليل خدمات شركة الكهرباء الاردنية.pdf", "category": "Services"}
        ]
    }

    all_raw_content = []

    # Process webpages
    for item in sources_to_process["webpages"]:
        print(f"Scraping Webpage: {item['url']}")
        content = scrape_webpage(item["url"])
        if content:
            all_raw_content.append({
                "source": item["url"],
                "category": item["category"],
                "content": content
            })

    # Process local PDFs
    for item in sources_to_process["local_pdfs"]:
        print(f"Scraping PDF: {item['path']}")
        content = scrape_local_pdf(item["path"])
        if content:
            all_raw_content.append({
                "source": item["path"], # Use path as the source identifier
                "category": item["category"],
                "content": content
            })

    # Save all scraped content to one file
    output_filename = 'raw_content.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_raw_content, f, ensure_ascii=False, indent=4)
        
    print(f"\nScraping complete. All raw text saved to '{output_filename}'")

if __name__ == "__main__":
    main()