# -*- coding: utf-8 -*-
"""
Utility functions for downloading content from URLs found by the search CLI.
Requires 'requests' library: pip install requests
"""
import os
import re
import sys
import requests
from urllib.parse import urlparse

def sanitize_filename(name):
    """Removes or replaces characters unsafe for filenames."""
    # Remove path separators and control characters
    name = re.sub(r'[\\/*?:"<>|\x00-\x1f]', '', name)
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    # Limit length
    max_len = 150
    if len(name) > max_len:
        name = name[:max_len]
    # Ensure it's not empty or just dots
    if not name or name.strip('.') == '':
        name = "downloaded_file"
    return name

def download_content(url, download_dir, filename_prefix=""):
    """
    Downloads content from a URL, saving as HTML or PDF.

    Args:
        url (str): The URL to download.
        download_dir (str): The directory to save the file in.
        filename_prefix (str, optional): A prefix for the filename (e.g., from title). Defaults to "".

    Returns:
        bool: True if download was successful or skipped, False otherwise.
    """
    if not url or not url.startswith(('http://', 'https://')):
        print(f"[!] Skipping invalid URL: {url}", file=sys.stderr)
        return True # Skipped, not technically an error

    print(f"[*] Attempting download: {url}", file=sys.stderr)
    try:
        # Make directory if it doesn't exist
        os.makedirs(download_dir, exist_ok=True)

        # Use a reasonable timeout
        headers = {'User-Agent': 'CLISearchEngine/1.0 (Download Utility)'} # Simple UA
        response = requests.get(url, headers=headers, timeout=20, stream=True, allow_redirects=True)
        response.raise_for_status() # Check for HTTP errors

        content_type = response.headers.get('content-type', '').lower()
        is_pdf = 'application/pdf' in content_type or url.lower().endswith('.pdf')

        # Determine filename
        if filename_prefix:
            base_name = sanitize_filename(filename_prefix)
        else:
            # Fallback: use sanitized path from URL
            parsed_url = urlparse(url)
            path_part = parsed_url.path.strip('/')
            if not path_part: path_part = parsed_url.netloc # Use domain if path is empty
            base_name = sanitize_filename(path_part)

        extension = ".pdf" if is_pdf else ".html"
        filename = f"{base_name}{extension}"
        filepath = os.path.join(download_dir, filename)

        # Avoid overwriting - add number if file exists
        counter = 1
        while os.path.exists(filepath):
            filename = f"{base_name}_{counter}{extension}"
            filepath = os.path.join(download_dir, filename)
            counter += 1

        # Download and save
        print(f"[*] Saving to: {filepath}", file=sys.stderr)
        with open(filepath, 'wb') as f:
            if is_pdf:
                # Save raw bytes for PDF
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            else:
                # Save decoded text for HTML (assuming UTF-8, requests usually handles this)
                # Ensure content is written as bytes
                try:
                    f.write(response.content) # Use response.content which is bytes
                except Exception as write_err:
                     print(f"[!] Error writing content bytes for {url}: {write_err}", file=sys.stderr)
                     # Attempt to write text with utf-8 encoding as fallback
                     try:
                         f.write(response.text.encode('utf-8', errors='replace'))
                     except Exception as fallback_write_err:
                          print(f"[!] Fallback write failed for {url}: {fallback_write_err}", file=sys.stderr)
                          return False


        print(f"[*] Successfully downloaded: {url}", file=sys.stderr)
        return True

    except requests.exceptions.RequestException as e:
        print(f"[!] Download Error for {url}: {e}", file=sys.stderr)
        return False
    except OSError as e:
        print(f"[!] Filesystem Error for {filepath}: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[!] Unexpected Error downloading {url}: {e}", file=sys.stderr)
        return False

if __name__ == '__main__':
    # Example Usage (for testing)
    test_dir = "./TestDownloads"
    print(f"Testing downloads to {test_dir}")
    download_content("https://example.com", test_dir, "example_page")
    # download_content("https://www.google.com", test_dir, "google_search") # Might get blocked
    # download_content("https://arxiv.org/pdf/2303.10130.pdf", test_dir, "arxiv_paper") # Example PDF
    print("Test complete.")