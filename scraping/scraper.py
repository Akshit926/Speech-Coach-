"""
Web Scraping Module for ToastMasterAI
Collects publicly available Toastmasters-related text from permitted public webpages.

IMPORTANT: This module only scrapes publicly available content.
Do not scrape private, login-protected, copyrighted full speech transcripts,
or restricted content.
"""

import os
import csv
import time
import requests
from bs4 import BeautifulSoup


# Default User-Agent
HEADERS = {
    'User-Agent': 'ToastMasterAI-Academic-Project/1.0 (Student NLP Project)'
}

# Timeout for requests
TIMEOUT = 10


def scrape_page(url: str) -> dict:
    """
    Scrape text content from a public webpage.
    
    Args:
        url: The URL to scrape
    
    Returns:
        Dictionary with title, text, paragraphs, word_count
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Get title
        title = soup.title.string.strip() if soup.title and soup.title.string else 'Untitled'
        
        # Get all paragraphs
        paragraphs = []
        for p in soup.find_all('p'):
            text = p.get_text(strip=True)
            if len(text) > 20:  # Filter out very short paragraphs
                paragraphs.append(text)
        
        full_text = ' '.join(paragraphs)
        word_count = len(full_text.split())
        
        return {
            'success': True,
            'url': url,
            'title': title,
            'text': full_text,
            'paragraphs': paragraphs,
            'word_count': word_count,
            'num_paragraphs': len(paragraphs)
        }
    
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'url': url,
            'error': str(e),
            'title': '', 'text': '', 'paragraphs': [],
            'word_count': 0, 'num_paragraphs': 0
        }


def scrape_multiple_pages(urls: list, delay: float = 1.0) -> list:
    """
    Scrape multiple pages with a polite delay between requests.
    
    Args:
        urls: List of URLs to scrape
        delay: Seconds to wait between requests (be polite!)
    
    Returns:
        List of scrape result dictionaries
    """
    results = []
    for i, url in enumerate(urls):
        print(f"  Scraping [{i+1}/{len(urls)}]: {url}")
        result = scrape_page(url)
        results.append(result)
        
        if i < len(urls) - 1:
            time.sleep(delay)
    
    return results


def save_scraped_data(results: list, filepath: str = None):
    """
    Save scraped data to CSV.
    
    Args:
        results: List of scrape result dictionaries
        filepath: Output CSV file path
    """
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, 'data', 'scraped_data.csv')
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    successful = [r for r in results if r.get('success')]
    
    if not successful:
        print("No successful scrapes to save.")
        return
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['url', 'title', 'text', 'word_count'])
        writer.writeheader()
        for r in successful:
            writer.writerow({
                'url': r['url'],
                'title': r['title'],
                'text': r['text'],
                'word_count': r['word_count']
            })
    
    print(f"Saved {len(successful)} records to {filepath}")


def clean_scraped_text(text: str) -> str:
    """
    Clean scraped text for NLP analysis.
    
    Args:
        text: Raw scraped text
    
    Returns:
        Cleaned text
    """
    import re
    
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove common web artifacts
    text = re.sub(r'Cookie\s*Policy.*?(?:\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Privacy\s*Policy.*?(?:\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Subscribe.*?newsletter.*?(?:\.|$)', '', text, flags=re.IGNORECASE)
    
    return text.strip()


# ─── Example URLs for academic demonstration ─────────────────
# These are public, non-restricted pages about public speaking tips
SAMPLE_URLS = [
    'https://en.wikipedia.org/wiki/Toastmasters_International',
    'https://en.wikipedia.org/wiki/Public_speaking',
    'https://en.wikipedia.org/wiki/Rhetoric',
]


if __name__ == '__main__':
    print("=" * 60)
    print("ToastMasterAI — Web Scraping Module")
    print("=" * 60)
    print("\nScraping public Wikipedia pages for supplementary NLP data...")
    print("Note: Only publicly available content is collected.\n")
    
    results = scrape_multiple_pages(SAMPLE_URLS)
    
    successful = [r for r in results if r['success']]
    print(f"\nSuccessfully scraped: {len(successful)}/{len(SAMPLE_URLS)} pages")
    
    for r in successful:
        print(f"  - {r['title']}: {r['word_count']} words")
    
    save_scraped_data(results)
    print("\nScraping complete!")
