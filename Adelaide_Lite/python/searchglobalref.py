#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import subprocess
import time
import json

# --- Environment Setup ---
def apply_base_env():
    """Load core environment variables from config.json to ensure consistent execution."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                base_env = config.get("base_env", {})
                for key, value in base_env.items():
                    os.environ[key] = value
        except Exception as e:
            print(f"⚠️ Error loading base_env: {e}", file=sys.stderr)

# --- Bootstrap Virtual Environment ---
VENV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pyvenv")
REQUIREMENTS = ["selenium", "webdriver-manager", "beautifulsoup4", "ocrmypdf", "numpy", "requests", "Pillow"]

def bootstrap_venv():
    """Ensures the script runs in its dedicated virtual environment."""
    apply_base_env()
    venv_abs = os.path.abspath(VENV_DIR)
    
    # If not in the correct venv, ensure it exists and switch to it
    if os.path.abspath(sys.prefix) != venv_abs:
        if not os.path.exists(VENV_DIR):
            print(f"[*] Creating virtual environment in {VENV_DIR}...", file=sys.stderr)
            subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)
            
        if os.name == 'nt':
            python_exe = os.path.join(VENV_DIR, "Scripts", "python.exe")
        else:
            python_exe = os.path.join(VENV_DIR, "bin", "python")
        
        if os.path.exists(python_exe):
            os.execv(python_exe, [python_exe] + sys.argv)

    # Once inside the venv, verify requirements
    try:
        import selenium
        import webdriver_manager
        import bs4
        import numpy
        import requests
        import PIL.Image
    except ImportError:
        print(f"[*] Missing dependencies. Installing: {', '.join(REQUIREMENTS)}...", file=sys.stderr)
        if os.name == 'nt':
            pip_exe = os.path.join(VENV_DIR, "Scripts", "pip.exe")
        else:
            pip_exe = os.path.join(VENV_DIR, "bin", "pip")
        subprocess.run([pip_exe, "install", "--upgrade", "pip"], check=True)
        subprocess.run([pip_exe, "install"] + REQUIREMENTS, check=True)
        # Re-execute one last time to pick up new packages
        os.execv(sys.executable, [sys.executable] + sys.argv)

bootstrap_venv()

# --- Post-Bootstrap Imports ---
import select
import random
import re
from urllib.parse import urlparse, parse_qs, quote_plus, urljoin
from datetime import datetime
import base64
from bs4 import BeautifulSoup 
import io
from PIL import Image

# --- Selenium Imports ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement 
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException, InvalidSelectorException

# --- Ollama Configuration ---
OLLAMA_BASE_URL = os.environ.get("OLLAMA_PROXY_URL", "http://localhost:11435")
OLLAMA_EMBED_ENDPOINT = f"{OLLAMA_BASE_URL}/api/embed"
OLLAMA_MODEL = "qwen3-embedding:0.6b"

# --- User Agents ---
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/109.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/109.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 13_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15',
]

# --- Helper Functions ---

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def save_as_pdf(driver: WebDriver, title: str):
    """Saves the current page as a PDF in ~/Downloads with OCR and compression (silently)."""
    try:
        safe_title = "".join([c for c in title if c.isalnum() or c in (' ', '.', '_')]).strip()
        filename = f"{safe_title[:100]}.pdf"
        final_path = os.path.expanduser(f"~/Downloads/{filename}")
        temp_raw_path = final_path + ".raw.pdf"

        print_options = {'landscape': False, 'displayHeaderFooter': False, 'printBackground': True, 'preferCSSPageSize': True}
        result = driver.execute_cdp_cmd("Page.printToPDF", print_options)
        
        with open(temp_raw_path, "wb") as f:
            f.write(base64.b64decode(result['data']))
        
        try:
            import ocrmypdf
            ocrmypdf.ocr(temp_raw_path, final_path, optimize=2, skip_text=True, progress_bar=False)
            if os.path.exists(temp_raw_path): os.remove(temp_raw_path)
            return final_path
        except Exception:
            if os.path.exists(final_path): os.remove(final_path)
            os.rename(temp_raw_path, final_path)
            return final_path
    except Exception:
        return None

def capture_base64_screenshot(driver: WebDriver):
    """Captures a screenshot, compresses it (quality 27), and returns base64 string."""
    try:
        screenshot_png = driver.get_screenshot_as_png()
        img = Image.open(io.BytesIO(screenshot_png))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        
        # Resize to max 512p
        img.thumbnail((512, 512))
        
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=10)
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        print(f"⚠️ Screenshot failed: {e}", file=sys.stderr)
        return None

def extract_web_images(driver: WebDriver, max_images=3):
    """Finds images on the page, downloads, resizes to 512p, and returns list of base64 strings."""
    base64_images = []
    try:
        # Find img tags with src
        imgs = driver.find_elements(By.TAG_NAME, "img")
        count = 0
        for img_el in imgs:
            if count >= max_images:
                break
            src = img_el.get_attribute("src")
            if not src or not src.startswith("http"):
                continue
            
            try:
                # Download image
                resp = requests.get(src, timeout=5, stream=True)
                if resp.status_code == 200:
                    img = Image.open(io.BytesIO(resp.content))
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    
                    # Resize to max 512p
                    img.thumbnail((512, 512))
                    
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG", quality=27)
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    base64_images.append(f"data:image/jpeg;base64,{img_str}")
                    count += 1
            except Exception:
                continue
    except Exception as e:
        print(f"⚠️ Image extraction failed: {e}", file=sys.stderr)
    return base64_images

def generate_apa7_reference(title, url):
    today = datetime.now().strftime("%Y, %B %d")
    clean_title = title.strip().rstrip('.')
    # Use online fetch date only, avoiding guessing publication date
    return f"{clean_title}. (Fetched: {today}). {url}"

def extract_page_snippet(driver: WebDriver, max_chars=400):
    try:
        paragraphs = driver.find_elements(By.TAG_NAME, "p")
        text_content = []
        for p in paragraphs:
            text = p.text.strip()
            if len(text) > 20:
                text_content.append(text)
            if len(" ".join(text_content)) > max_chars: break
        
        full_text = " ".join(text_content)
        if not full_text: full_text = driver.find_element(By.TAG_NAME, "body").text.strip()
        return full_text[:max_chars].replace("\n", " ") + "..." if len(full_text) > max_chars else full_text
    except Exception: return None

def send_notification(title, message):
    try:
        # Escape double quotes for osascript
        safe_message = message.replace('"', '\\"')
        safe_title = title.replace('"', '\\"')
        script = f'display notification "{safe_message}" with title "{safe_title}" sound name "Glass"'
        subprocess.run(["osascript", "-e", script])
    except Exception as e: print(f"[!] Failed to send notification: {e}", file=sys.stderr)

def check_for_captcha(driver: WebDriver, timeout_seconds=60):
    solvable_indicators = ["captcha", "hcaptcha", "recaptcha", "verify you are human", "challenge"]
    hard_block_indicators = ["access denied", "403 forbidden", "automated access", "bot detection", "unusual traffic", "sorry for the inconvenience", "blocked", "automated queries"]
    
    def get_status():
        try:
            src = driver.page_source.lower()
            ttl = driver.title.lower()
            url = driver.current_url.lower()
            
            # Check for actual CAPTCHA widgets or challenges
            is_solvable = any(i in src for i in solvable_indicators) or any(i in ttl for i in solvable_indicators)
            
            # Check for "Access Denied" or "Blocked" messages
            is_blocked = any(i in src for i in hard_block_indicators) or any(i in ttl for i in hard_block_indicators) or "google.com/sorry/" in url
            
            return is_solvable, is_blocked
        except: return False, False

    is_solvable, is_blocked = get_status()
    
    if is_solvable or is_blocked:
        site = driver.current_url
        if is_blocked and not is_solvable:
            print(f"\n" + "!"*60 + f"\n[!!!] HARD BLOCK DETECTED: {site}\nNo interactive CAPTCHA found to solve (Access Denied).\n" + "!"*60 + "\n", file=sys.stderr)
            send_notification("Site Blocked", f"Access Denied on {site}. No solver available.")
            return False # Skip waiting as there is nothing to solve

        send_notification("Searcher Attention Needed", f"CAPTCHA triggered on {site}. Please solve.")
        print("\n" + "!"*60 + "\n[!!!] BOT DETECTION TRIGGERED (CAPTCHA SOLVER NEEDED)\n" + "!"*60 + "\n", file=sys.stderr)
        
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            is_solvable, is_blocked = get_status()
            if not is_solvable and not is_blocked:
                print("[*] Challenge solved. Resuming...", file=sys.stderr)
                return True
            
            # Allow manual override via terminal
            rlist, _, _ = select.select([sys.stdin], [], [], 0.5)
            if rlist:
                sys.stdin.readline()
                return True
            time.sleep(1)
        return False
    return True # NO CAPTCHA/BLOCK FOUND - OK TO PROCEED

def extract_pdf_link(block_element: WebElement) -> str | None:
    pdf_selector = 'div.gs_ggsd a, a[href$=".pdf"], a[href*=".pdf?"]'
    try:
        tag = block_element.find_element(By.CSS_SELECTOR, pdf_selector)
        href = tag.get_attribute('href')
        if href and href.startswith('http'): return href
    except: pass
    return None

def ensure_ollama_running():
    import requests
    try:
        requests.get(f"{OLLAMA_BASE_URL}", timeout=2)
        print(f"✅ Ollama reachable at {OLLAMA_BASE_URL}", file=sys.stderr)
        return True
    except:
        print(f"⚠️ Ollama not reachable. Attempting restart...", file=sys.stderr)
        subprocess.run(["launchctl", "setenv", "OLLAMA_HOST", "0.0.0.0:1234"], check=False)
        subprocess.run(["brew", "services", "restart", "ollama"], check=False)
        time.sleep(3)
        try:
            requests.get(f"{OLLAMA_BASE_URL}", timeout=2)
            return True
        except: return False

def get_embedding(text: str):
    import requests
    import numpy as np
    if not text: return None
    try:
        resp = requests.post(OLLAMA_EMBED_ENDPOINT, json={"model": OLLAMA_MODEL, "input": text}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "embeddings" in data and len(data["embeddings"]) > 0:
            return np.array(data["embeddings"][0])
        elif "embedding" in data:
            return np.array(data["embedding"])
        return None
    except: return None

# --- Searching Functions ---

def search_duckduckgo(driver: WebDriver, query, num_results, timeout, max_pages=1):
    print(f"[*] Searching DuckDuckGo for '{query}'...", file=sys.stderr)
    results = []
    try:
        driver.get(f"https://duckduckgo.com/?q={quote_plus(query)}&ia=web")
    except TimeoutException: return results
    if not check_for_captcha(driver, timeout_seconds=timeout): return results
    processed_urls = set()
    for page in range(max_pages):
        try:
            blocks = driver.find_elements(By.CSS_SELECTOR, "article[data-testid='result']")
            for block in blocks:
                if len(results) >= num_results: break
                try:
                    title_tag = block.find_element(By.CSS_SELECTOR, "a[data-testid='result-title-a']")
                    url = title_tag.get_attribute('href')
                    if url and "duckduckgo.com/y.js" in url:
                        try: url = block.find_element(By.CSS_SELECTOR, "a.result__url").get_attribute('href')
                        except: pass
                    if not url or url in processed_urls: continue
                    snippet = ""
                    try: snippet = block.find_element(By.CSS_SELECTOR, "div[data-testid='result-snippet']").text
                    except: pass
                    results.append({'title': title_tag.text, 'url': url, 'snippet': snippet})
                    processed_urls.add(url)
                except: continue
            if len(results) >= num_results: break
            if page < max_pages - 1:
                try:
                    driver.find_element(By.ID, "more-results").click()
                    time.sleep(2)
                except: break
        except: break
    return results

def search_google(driver: WebDriver, query, num_results, timeout, max_pages=1):
    results = []
    processed_urls = set()
    for page in range(max_pages):
        start = page * 10
        try:
            driver.get(f"https://www.google.com/search?q={quote_plus(query)}&start={start}")
        except TimeoutException:
            print(f"[*] Google timeout on page {page}", file=sys.stderr)
            break
        if not check_for_captcha(driver, timeout_seconds=timeout): 
            break
        blocks = driver.find_elements(By.CSS_SELECTOR, "div.g, div.kvH3mc")
        if not blocks: break
        for block in blocks:
            if len(results) >= num_results: break
            try:
                h3 = block.find_element(By.TAG_NAME, "h3")
                link = block.find_element(By.TAG_NAME, "a").get_attribute('href')
                if not link or "google.com" in link or link in processed_urls: continue
                results.append({'title': h3.text, 'url': link, 'snippet': block.text[:200]})
                processed_urls.add(link)
            except: continue
    return results

def search_searx(driver: WebDriver, instance_url, query, num_results, timeout):
    results = []
    driver.get(f"{instance_url}/search?q={quote_plus(query)}")
    check_for_captcha(driver)
    blocks = driver.find_elements(By.CSS_SELECTOR, 'div.result, article.result')
    for block in blocks:
        if len(results) >= num_results: break
        try:
            a = block.find_element(By.TAG_NAME, "a")
            url = a.get_attribute('href')
            results.append({'title': a.text, 'url': url, 'snippet': block.text[:200]})
        except: continue
    return results

def search_semantic_scholar(driver: WebDriver, query, num_results, timeout):
    results = []
    try:
        driver.get(f"https://www.semanticscholar.org/search?q={quote_plus(query)}")
    except TimeoutException: return results
    if not check_for_captcha(driver, timeout_seconds=timeout): return results
    blocks = driver.find_elements(By.CSS_SELECTOR, 'div[data-test-id="search-result-card"]')
    for block in blocks:
        if len(results) >= num_results: break
        try:
            a = block.find_element(By.CSS_SELECTOR, 'a[data-test-id="title-link"]')
            results.append({'title': a.text, 'url': a.get_attribute('href'), 'snippet': block.text[:200]})
        except: continue
    return results

def search_google_scholar(driver, query, num_results, timeout, max_pages=1):
    results = []
    for page in range(max_pages):
        try:
            driver.get(f"https://scholar.google.com/scholar?q={quote_plus(query)}&start={page*10}")
        except TimeoutException:
            print(f"[*] Scholar timeout on page {page}", file=sys.stderr)
            break
        if not check_for_captcha(driver, timeout_seconds=timeout): 
            break
        blocks = driver.find_elements(By.CSS_SELECTOR, 'div.gs_r.gs_or.gs_scl')
        if not blocks: break
        for block in blocks:
            if len(results) >= num_results: break
            try:
                a = block.find_element(By.CSS_SELECTOR, 'h3.gs_rt a')
                results.append({'title': a.text, 'url': a.get_attribute('href'), 'snippet': block.text[:200]})
            except: continue
    return results

def search_base(driver, query, num_results, timeout):
    results = []
    try:
        driver.get(f"https://www.base-search.net/Search/Results?lookfor={quote_plus(query)}")
    except TimeoutException: return results
    if not check_for_captcha(driver, timeout_seconds=timeout): return results
    blocks = driver.find_elements(By.CSS_SELECTOR, 'div.record')
    for block in blocks:
        if len(results) >= num_results: break
        try:
            a = block.find_element(By.CSS_SELECTOR, 'a.title')
            results.append({'title': a.text, 'url': a.get_attribute('href'), 'snippet': block.text[:200]})
        except: continue
    return results

def search_core(driver, query, num_results, timeout):
    results = []
    try:
        driver.get(f"https://core.ac.uk/search?q={quote_plus(query)}")
    except TimeoutException: return results
    if not check_for_captcha(driver, timeout_seconds=timeout): return results
    blocks = driver.find_elements(By.CSS_SELECTOR, 'div.result-item')
    for block in blocks:
        if len(results) >= num_results: break
        try:
            a = block.find_element(By.TAG_NAME, "a")
            results.append({'title': a.text, 'url': a.get_attribute('href'), 'snippet': block.text[:200]})
        except: continue
    return results

def search_sciencegov(driver, query, num_results, timeout):
    results = []
    try:
        driver.get(f"https://www.science.gov/scigov/desktop/en/results.html?q={quote_plus(query)}")
    except TimeoutException: return results
    if not check_for_captcha(driver, timeout_seconds=timeout): return results
    time.sleep(3)
    blocks = driver.find_elements(By.CSS_SELECTOR, 'div.result')
    for block in blocks:
        if len(results) >= num_results: break
        try:
            a = block.find_element(By.CSS_SELECTOR, 'div.title > a')
            results.append({'title': a.text, 'url': a.get_attribute('href'), 'snippet': block.text[:200]})
        except: continue
    return results

def search_baidu_scholar(driver, query, num_results, timeout):
    results = []
    try:
        driver.get(f"https://xueshu.baidu.com/s?wd={quote_plus(query)}")
    except TimeoutException: return results
    if not check_for_captcha(driver, timeout_seconds=timeout): return results
    blocks = driver.find_elements(By.CSS_SELECTOR, 'div.result.sc_default_result')
    for block in blocks:
        if len(results) >= num_results: break
        try:
            a = block.find_element(By.TAG_NAME, "a")
            results.append({'title': a.text, 'url': a.get_attribute('href'), 'snippet': block.text[:200]})
        except: continue
    return results

def search_refseek(driver, query, num_results, timeout):
    results = []
    try:
        driver.get(f"https://www.refseek.com/search?q={quote_plus(query)}")
    except TimeoutException: return results
    if not check_for_captcha(driver, timeout_seconds=timeout): return results
    blocks = driver.find_elements(By.CSS_SELECTOR, 'div.gsc-result')
    for block in blocks:
        if len(results) >= num_results: break
        try:
            a = block.find_element(By.CSS_SELECTOR, 'a.gs-title')
            results.append({'title': a.text, 'url': a.get_attribute('href'), 'snippet': block.text[:200]})
        except: continue
    return results

def search_sciencedirect(driver, query, num_results, timeout):
    results = []
    try:
        driver.get(f"https://www.sciencedirect.com/search?qs={quote_plus(query)}")
    except TimeoutException: return results
    if not check_for_captcha(driver, timeout_seconds=timeout): return results
    blocks = driver.find_elements(By.CSS_SELECTOR, 'li.ResultItem')
    for block in blocks:
        if len(results) >= num_results: break
        try:
            a = block.find_element(By.CSS_SELECTOR, 'a.result-list-title-link')
            results.append({'title': a.text, 'url': a.get_attribute('href'), 'snippet': block.text[:200]})
        except: continue
    return results

def search_mdpi(driver, query, num_results, timeout):
    results = []
    try:
        driver.get(f"https://www.mdpi.com/search?q={quote_plus(query)}")
    except TimeoutException: return results
    if not check_for_captcha(driver, timeout_seconds=timeout): return results
    blocks = driver.find_elements(By.CSS_SELECTOR, 'article.article-item')
    for block in blocks:
        if len(results) >= num_results: break
        try:
            a = block.find_element(By.CSS_SELECTOR, 'a.title-link')
            results.append({'title': a.text, 'url': a.get_attribute('href'), 'snippet': block.text[:200]})
        except: continue
    return results

def search_tandf(driver, query, num_results, timeout):
    results = []
    try:
        driver.get(f"https://www.tandfonline.com/action/doSearch?AllField={quote_plus(query)}")
    except TimeoutException: return results
    if not check_for_captcha(driver, timeout_seconds=timeout): return results
    blocks = driver.find_elements(By.CSS_SELECTOR, 'div.searchResultItem')
    for block in blocks:
        if len(results) >= num_results: break
        try:
            a = block.find_element(By.CSS_SELECTOR, 'a.hlFld-Title')
            results.append({'title': a.text, 'url': a.get_attribute('href'), 'snippet': block.text[:200]})
        except: continue
    return results

def search_ieee(driver, query, num_results, timeout):
    results = []
    try:
        driver.get(f"https://ieeexplore.ieee.org/search/searchresult.jsp?queryText={quote_plus(query)}")
    except TimeoutException: return results
    if not check_for_captcha(driver, timeout_seconds=timeout): return results
    time.sleep(3)
    blocks = driver.find_elements(By.CSS_SELECTOR, 'xpl-results-item')
    for block in blocks:
        if len(results) >= num_results: break
        try:
            a = block.find_element(By.CSS_SELECTOR, 'h2 a')
            results.append({'title': a.text, 'url': a.get_attribute('href'), 'snippet': block.text[:200]})
        except: continue
    return results

def search_aiaa(driver, query, num_results, timeout):
    results = []
    try:
        driver.get(f"https://arc.aiaa.org/action/doSearch?AllField={quote_plus(query)}")
    except TimeoutException: return results
    if not check_for_captcha(driver, timeout_seconds=timeout): return results
    blocks = driver.find_elements(By.CSS_SELECTOR, 'div.search-item')
    for block in blocks:
        if len(results) >= num_results: break
        try:
            a = block.find_element(By.CSS_SELECTOR, 'h3.item__title a')
            results.append({'title': a.text, 'url': a.get_attribute('href'), 'snippet': block.text[:200]})
        except: continue
    return results

def search_ncbi(driver, query, num_results, timeout):
    results = []
    try:
        driver.get(f"https://pubmed.ncbi.nlm.nih.gov/?term={quote_plus(query)}")
    except TimeoutException: return results
    if not check_for_captcha(driver, timeout_seconds=timeout): return results
    blocks = driver.find_elements(By.CSS_SELECTOR, 'article.full-docsum')
    for block in blocks:
        if len(results) >= num_results: break
        try:
            a = block.find_element(By.CSS_SELECTOR, 'a.docsum-title')
            results.append({'title': a.text, 'url': a.get_attribute('href'), 'snippet': block.text[:200]})
        except: continue
    return results

def search_springer(driver, query, num_results, timeout):
    results = []
    try:
        driver.get(f"https://link.springer.com/search?query={quote_plus(query)}")
    except TimeoutException: return results
    if not check_for_captcha(driver, timeout_seconds=timeout): return results
    blocks = driver.find_elements(By.CSS_SELECTOR, 'li.results-list__item')
    for block in blocks:
        if len(results) >= num_results: break
        try:
            a = block.find_element(By.CSS_SELECTOR, 'h2 a')
            results.append({'title': a.text, 'url': a.get_attribute('href'), 'snippet': block.text[:200]})
        except: continue
    return results

def store_in_memory(content, ollama_external=None):
    """Invokes memorythoughts.py to store content."""
    try:
        memory_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memorythoughts.py")
        cmd = [sys.executable, memory_script, "--string", content]
        if ollama_external:
            cmd.extend(["--ollamaHost", ollama_external])
        subprocess.run(cmd, check=False)
    except Exception as e:
        print(f"⚠️ Failed to store memory: {e}", file=sys.stderr)

def main():
    import argparse
    import json
    import numpy as np
    
    ALLOWED = ['ddg', 'google', 'searx', 'sem', 'scholar', 'base', 'core', 'scigov', 'baidu', 'refseek', 'scidirect', 'mdpi', 'tandf', 'ieee', 'springer', 'aiaa', 'ncbi']
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--engines", nargs='*', default=['all'])
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--jsonIO", action="store_true", help="Output results in JSON format.")
    parser.add_argument("--ollamaExternal", type=str, default=None, help="Custom Ollama server address.")
    parser.add_argument("--ollamaHost", type=str, default=None, help="Custom Ollama host address.")
    args = parser.parse_args()

    global OLLAMA_BASE_URL, OLLAMA_EMBED_ENDPOINT
    host = args.ollamaHost or args.ollamaExternal
    if host:
        OLLAMA_BASE_URL = host if host.startswith("http") else f"http://{host}"
        OLLAMA_EMBED_ENDPOINT = f"{OLLAMA_BASE_URL}/api/embed"

    selected = ALLOWED if 'all' in args.engines else [e for e in args.engines if e in ALLOWED]
    engine_map = {
        'ddg': search_duckduckgo, 'google': search_google, 'searx': search_searx, 'sem': search_semantic_scholar,
        'scholar': search_google_scholar, 'base': search_base, 'core': search_core, 'scigov': search_sciencegov,
        'baidu': search_baidu_scholar, 'refseek': search_refseek, 'scidirect': search_sciencedirect,
        'mdpi': search_mdpi, 'tandf': search_tandf, 'ieee': search_ieee, 'springer': search_springer,
        'aiaa': search_aiaa, 'ncbi': search_ncbi
    }

    if args.jsonIO:
        print(json.dumps({"phase": 1, "status": "start", "query": args.query}), flush=True)

    ollama_ready = ensure_ollama_running()
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    
    all_flat = []
    with webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options) as driver:
        driver.set_page_load_timeout(args.timeout)
        for eng in selected:
            if not args.jsonIO:
                print(f"[*] Searching {eng.upper()}...", file=sys.stderr)
            s_args = [driver, args.query, args.num, args.timeout]
            if eng in ['ddg', 'google', 'scholar']: s_args.append(args.pages)
            elif eng == 'searx': s_args.insert(1, "https://search.inetol.net")
            
            try:
                results = engine_map[eng](*s_args)
                for r in results:
                    r['source_engine'] = eng
                    r['apa7_reference'] = generate_apa7_reference(r['title'], r['url'])
                    all_flat.append(r)
            except Exception as e: 
                if not args.jsonIO:
                    print(f"Error {eng}: {e}", file=sys.stderr)

        if args.jsonIO:
            print(json.dumps({"phase": 1, "status": "complete", "results": all_flat}), flush=True)
            print(json.dumps({"phase": 2, "status": "start"}), flush=True)

        for r in all_flat:
            if not args.jsonIO:
                print(f"[*] Visiting: {r['url']} (it may be Invalid trigger)", file=sys.stderr)
            try:
                driver.get(r['url'])
                check_for_captcha(driver)
                time.sleep(2)
                r['snippet'] = extract_page_snippet(driver)
                # Save PDF silently for archival
                save_as_pdf(driver, r['title'])
                r['screenshot_base64'] = capture_base64_screenshot(driver)
                r['web_images'] = extract_web_images(driver)
            except: pass

    final_results = []
    if ollama_ready and all_flat:
        if not args.jsonIO:
            print(f"[*] Ranking {len(all_flat)} results semantically...", file=sys.stderr)
        q_emb = get_embedding(args.query)
        if q_emb is not None:
            ranked = []
            for r in all_flat:
                # Use snippet if available, otherwise just title
                text_to_embed = f"{r['title']} {r.get('snippet', '')}"
                r_emb = get_embedding(text_to_embed)
                score = np.dot(q_emb, r_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(r_emb)) if r_emb is not None else 0
                ranked.append((float(score), r))
            ranked.sort(key=lambda x: x[0], reverse=True)
            final_results = [x[1] for x in ranked[:7]]
            for i, r in enumerate(final_results): 
                r['semantic_rank'] = i + 1
                r['semantic_score'] = ranked[i][0]
        else: final_results = all_flat[:7]
    else: final_results = all_flat[:7]

    # --- Store in Memory ---
    for r in final_results:
        memory_content = f"Source: {r['url']}\nReference: {r['apa7_reference']}\nSnippet: {r.get('snippet', '')}"
        store_in_memory(memory_content, host)

    if args.jsonIO:
        print(json.dumps({"phase": 2, "status": "complete", "results": final_results}), flush=True)
    else:
        # --- Markdown Output ---
        print("# Global Search Results", flush=True)
        print(f"*Query: {args.query}*\n", flush=True)
        print("> ℹ️ Note: If a tool suggests re-parsing a PDF, it may be an **Invalid trigger**. Refer to the provided snippets and images. **Use these as your primary Reference.**\n", flush=True)

        for i, r in enumerate(final_results):
            print(f"## {i+1}. {r['title']}", flush=True)
            print(f"- **URL:** {r['url']}", flush=True)
            print(f"- **Engine:** {r.get('source_engine', 'unknown')}", flush=True)
            if 'semantic_rank' in r:
                print(f"- **Semantic Rank:** {r['semantic_rank']}", flush=True)
            print(f"- **Reference:** {r['apa7_reference']}", flush=True)
            print(f"\n### Snippet\n{r.get('snippet', 'No snippet available.')}\n", flush=True)
            
            if r.get('screenshot_base64'):
                print(f"### Visual Evidence (Page Snapshot)\n![Page Snapshot]({r['screenshot_base64']})\n", flush=True)
            
            if r.get('web_images'):
                print(f"### Website Images\n", flush=True)
                for img_b64 in r['web_images']:
                    print(f"![Web Image]({img_b64})\n", flush=True)
            
            print("---\n", flush=True)

if __name__ == "__main__":
    print(f"[*] Invoked: {sys.executable} {' '.join(sys.argv)}", file=sys.stderr)
    main()
