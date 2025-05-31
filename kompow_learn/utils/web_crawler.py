import requests
from bs4 import BeautifulSoup
import re

def fetch_url_content(url: str) -> str | None:
    """
    Fetches and extracts plain text content from a given URL.
    Removes common non-content tags like script, style, nav, footer, header.
    Returns None if fetching fails or no meaningful text is found.
    """
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url # Add scheme if missing, default to http

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # Check content type to avoid parsing non-HTML content like PDFs, images directly with BeautifulSoup
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type:
            print(f"Skipping URL {url} as content type is not HTML ({content_type}).")
            return None # Or handle differently, e.g., download if it's a file

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove common non-content tags
        tags_to_remove = ['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'meta', 'link']
        for tag_name in tags_to_remove:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Attempt to find a main content area if possible (very site-specific, basic example)
        main_content = soup.find('main') or soup.find('article') or soup.find(class_=re.compile("content|main|article|body"))
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            text = soup.get_text(separator='\n', strip=True)

        if not text.strip(): # Check if extracted text is empty
            print(f"No meaningful text found at {url} after parsing.")
            return None

        return text

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error fetching URL {url}: {e}")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error fetching URL {url}: {e}")
        return None
    except requests.exceptions.Timeout as e:
        print(f"Timeout fetching URL {url}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while processing URL {url}: {e}")
        return None

if __name__ == '__main__':
    test_urls = [
        "http://example.com",
        "https://www.google.com", # Might be heavily JS-driven or have security measures
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "invalid-url-without-scheme.com",
        "http://nonexistentdomain12345.com",
        "https://www.w3.org/TR/PNG/iso_8859-1.txt" # Non-HTML content
    ]

    for url_to_test in test_urls:
        print(f"\n--- Attempting to fetch content from: {url_to_test} ---")
        content = fetch_url_content(url_to_test)
        if content:
            print(f"--- Content from {url_to_test} (first 500 chars) ---")
            print(content[:500].strip() + ("..." if len(content) > 500 else ""))
        else:
            print(f"Failed to fetch or no meaningful content from {url_to_test}")
