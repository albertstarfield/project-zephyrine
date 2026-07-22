# network_internet_knowledge_fetcher.py

import asyncio
import subprocess
import json
import base64
import io
import time
import requests
import numpy as np
from urllib.parse import quote_plus
from loguru import logger
from typing import Optional, Dict, Any, List, Callable
from PIL import Image
from datetime import datetime

try:
    from priority_lock import ELP0, ELP1
except ImportError:
    ELP0, ELP1 = 0, 1

try:
    from playwright.async_api import async_playwright, Page, BrowserContext, TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    logger.critical("Playwright library not found. Please run 'pip install playwright' and 'playwright install'.")
    PLAYWRIGHT_AVAILABLE = False

# --- Configuration & Constants ---
OLLAMA_BASE_URL = "http://localhost:11435"
OLLAMA_EMBED_ENDPOINT = f"{OLLAMA_BASE_URL}/api/embed"
OLLAMA_MODEL = "qwen3-embedding:0.6b"

# --- Helper Functions ---

def send_notification(title: str, message: str):
    """Sends a macOS notification using osascript."""
    try:
        safe_message = message.replace('"', '\\"')
        safe_title = title.replace('"', '\\"')
        script = f'display notification "{safe_message}" with title "{safe_title}" sound name "Glass"'
        subprocess.run(["osascript", "-e", script], check=False)
    except Exception as e:
        logger.error(f"Failed to send notification: {e}")

async def check_for_captcha(page: Page, timeout_seconds: int = 60) -> bool:
    """Detects CAPTCHA/Blocks and notifies the user to solve them manually."""
    solvable_indicators = ["captcha", "hcaptcha", "recaptcha", "verify you are human", "challenge"]
    hard_block_indicators = ["access denied", "403 forbidden", "automated access", "bot detection", "unusual traffic"]
    
    async def get_status():
        try:
            content = (await page.content()).lower()
            title = (await page.title()).lower()
            url = page.url.lower()
            is_solvable = any(i in content for i in solvable_indicators) or any(i in title for i in solvable_indicators)
            is_blocked = any(i in content for i in hard_block_indicators) or any(i in title for i in hard_block_indicators) or "google.com/sorry/" in url
            return is_solvable, is_blocked
        except: return False, False

    is_solvable, is_blocked = await get_status()
    if is_solvable or is_blocked:
        site = page.url
        if is_blocked and not is_solvable:
            logger.warning(f"HARD BLOCK DETECTED: {site}")
            send_notification("Site Blocked", f"Access Denied on {site}. No solver available.")
            return False

        send_notification("Searcher Attention Needed", f"CAPTCHA triggered on {site}. Please solve in the browser.")
        logger.warning(f"BOT DETECTION TRIGGERED on {site}. Waiting for manual solution...")
        
        # In a headless environment, this wait might be futile unless the user can see it.
        # However, we wait for a bit to see if the state changes.
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            is_solvable, is_blocked = await get_status()
            if not is_solvable and not is_blocked:
                logger.info("Challenge solved. Resuming...")
                return True
            await asyncio.sleep(2)
        return False
    return True

async def capture_base64_screenshot(page: Page) -> Optional[str]:
    """Captures a screenshot, compresses it, and returns a base64 string."""
    try:
        screenshot_bytes = await page.screenshot(type="jpeg", quality=20)
        img = Image.open(io.BytesIO(screenshot_bytes))
        img.thumbnail((512, 512))
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=20)
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        logger.error(f"Screenshot failed: {e}")
        return None

async def extract_web_images(page: Page, max_images: int = 3) -> List[str]:
    """Extracts and resizes the top images from the page."""
    base64_images = []
    try:
        img_elements = await page.query_selector_all("img")
        count = 0
        for el in img_elements:
            if count >= max_images: break
            src = await el.get_attribute("src")
            if not src or not src.startswith("http"): continue
            try:
                resp = requests.get(src, timeout=5)
                if resp.status_code == 200:
                    img = Image.open(io.BytesIO(resp.content))
                    if img.mode in ("RGBA", "P"): img = img.convert("RGB")
                    img.thumbnail((512, 512))
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG", quality=27)
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    base64_images.append(f"data:image/jpeg;base64,{img_str}")
                    count += 1
            except: continue
    except Exception as e:
        logger.error(f"Image extraction failed: {e}")
    return base64_images

def get_embedding(text: str, embeddings_model: Any = None, priority: int = ELP0) -> Optional[np.ndarray]:
    """
    Fetches text embedding. Prefers the provided system embeddings_model with priority,
    falling back to direct Ollama API if no model is provided.
    """
    if not text: return None
    
    # Use system model if provided (AdelaideAlbertCortex pattern)
    if embeddings_model and hasattr(embeddings_model, "embed_query"):
        try:
            # Check if it accepts priority
            vector = embeddings_model.embed_query(text, priority=priority)
            return np.array(vector)
        except Exception as e:
            logger.error(f"System embedding call failed: {e}")
            # Fallback to Ollama below
            
    # Fallback/Direct Ollama call
    try:
        resp = requests.post(OLLAMA_EMBED_ENDPOINT, json={"model": OLLAMA_MODEL, "input": text}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "embeddings" in data and len(data["embeddings"]) > 0:
            return np.array(data["embeddings"][0])
        elif "embedding" in data:
            return np.array(data["embedding"])
    except Exception as e:
        logger.debug(f"Direct Ollama embedding fallback failed: {e}")
    return None

def generate_apa7_reference(title: str, url: str) -> str:
    """Generates a basic APA7-style reference string."""
    today = datetime.now().strftime("%Y, %B %d")
    return f"{title.strip().rstrip('.')}. (Fetched: {today}). {url}"

# --- Individual Scraper Functions ---
# Each function is designed to be resilient, but websites change their structure.
# These selectors may need updating over time.

async def _scrape_google_scholar(page: Page, query: str) -> List[Dict[str, Any]]:
    """Scraper for Google Scholar."""
    search_url = f"https://scholar.google.com/scholar?q={quote_plus(query)}"
    await page.goto(search_url, wait_until="domcontentloaded", timeout=20000)
    
    if await page.query_selector("#gs_captcha_f"):
        logger.error("Google Scholar CAPTCHA detected. Cannot proceed.")
        return []
        
    results = []
    # Selector for each result block
    result_blocks = await page.query_selector_all("div.gs_r.gs_or.gs_scl")
    for block in result_blocks:
        title_element = await block.query_selector("h3.gs_rt a")
        snippet_element = await block.query_selector("div.gs_rs")
        pdf_link_element = await block.query_selector("div.gs_ggsd a")

        if title_element:
            title = await title_element.inner_text()
            url = await title_element.get_attribute("href")
            snippet = await snippet_element.inner_text() if snippet_element else "No snippet available."
            pdf_url = await pdf_link_element.get_attribute("href") if pdf_link_element else None

            results.append({"title": title, "url": url, "snippet": snippet, "pdf_url": pdf_url})
    return results

async def _scrape_semantic_scholar(page: Page, query: str) -> List[Dict[str, Any]]:
    """Scraper for Semantic Scholar."""
    search_url = f"https://www.semanticscholar.org/search?q={quote_plus(query)}"
    await page.goto(search_url, wait_until="domcontentloaded", timeout=20000)
    await page.wait_for_selector('div[data-test-id="paper-card"]', timeout=15000)

    results = []
    # Selector for each result block
    result_blocks = await page.query_selector_all('div[data-test-id="paper-card"]')
    for block in result_blocks:
        title_element = await block.query_selector('a[data-test-id="title-link"]')
        snippet_element = await block.query_selector('span[data-test-id="abstract-truncated"]')

        if title_element:
            title = await title_element.inner_text()
            url = await title_element.get_attribute("href")
            if url and not url.startswith("http"):
                url = "https://www.semanticscholar.org" + url
            snippet = await snippet_element.inner_text() if snippet_element else "No snippet available."

            results.append({"title": title, "url": url, "snippet": snippet})
    return results

async def _scrape_core(page: Page, query: str) -> List[Dict[str, Any]]:
    """Scraper for CORE (core.ac.uk)."""
    search_url = f"https://core.ac.uk/search?q={quote_plus(query)}"
    await page.goto(search_url, wait_until="domcontentloaded", timeout=20000)

    results = []
    # Selector for each result block
    result_blocks = await page.query_selector_all('div[class^="result-item"]')
    for block in result_blocks:
        title_element = await block.query_selector("h2 a")
        # Snippet is in a 'p' tag that is a sibling of the 'div' containing the h2
        snippet_element = await block.query_selector("p")

        if title_element:
            title = await title_element.inner_text()
            url = await title_element.get_attribute("href")
            snippet = await snippet_element.inner_text() if snippet_element else "No snippet available."
            results.append({"title": title, "url": url, "snippet": snippet})
    return results

async def _scrape_sciencegov(page: Page, query: str) -> List[Dict[str, Any]]:
    """Scraper for Science.gov."""
    search_url = f"https://www.science.gov/scigov/desktop/en/results.html?q={quote_plus(query)}"
    await page.goto(search_url, wait_until="domcontentloaded", timeout=20000)

    results = []
    # Selector for each result block
    result_blocks = await page.query_selector_all("div.result")
    for block in result_blocks:
        title_element = await block.query_selector("div.title_and_url_holder > h3 > a")
        snippet_element = await block.query_selector("div.result_description")

        if title_element:
            title = await title_element.inner_text()
            url = await title_element.get_attribute("href")
            snippet = await snippet_element.inner_text() if snippet_element else "No snippet available."
            results.append({"title": title, "url": url, "snippet": snippet})
    return results

async def _scrape_baidu_scholar(page: Page, query: str) -> List[Dict[str, Any]]:
    """Scraper for Baidu Scholar (xueshu.baidu.com)."""
    search_url = f"https://xueshu.baidu.com/s?wd={quote_plus(query)}"
    await page.goto(search_url, wait_until="domcontentloaded", timeout=20000)

    results = []
    # Selector for each result block
    result_blocks = await page.query_selector_all("div.result.sc_default_result")
    for block in result_blocks:
        title_element = await block.query_selector("h3.t a")
        snippet_element = await block.query_selector("div.c_abstract")

        if title_element:
            title = await title_element.inner_text()
            url = await title_element.get_attribute("href")
            snippet = await snippet_element.inner_text() if snippet_element else "No snippet available."
            results.append({"title": title, "url": url, "snippet": snippet.strip()})
    return results

async def _scrape_refseek(page: Page, query:str) -> List[Dict[str, Any]]:
    """Scraper for RefSeek."""
    search_url = f"https://www.refseek.com/search?q={quote_plus(query)}"
    await page.goto(search_url, wait_until="domcontentloaded", timeout=20000)

    results = []
    # Refseek uses a Google Custom Search Engine (CSE)
    result_blocks = await page.query_selector_all("div.gsc-webResult.gsc-result")
    for block in result_blocks:
        title_element = await block.query_selector("a.gs-title")
        snippet_element = await block.query_selector("div.gs-bidi-start-align.gs-snippet")

        if title_element:
            title = await title_element.inner_text()
            url = await title_element.get_attribute("href")
            snippet = await snippet_element.inner_text() if snippet_element else "No snippet available."
            results.append({"title": title, "url": url, "snippet": snippet})
    return results

async def _scrape_sciencedirect(page: Page, query: str) -> List[Dict[str, Any]]:
    """Scraper for ScienceDirect."""
    search_url = f"https://www.sciencedirect.com/search?qs={quote_plus(query)}"
    await page.goto(search_url, wait_until="domcontentloaded", timeout=25000)
    await page.wait_for_selector("li.ResultItem", timeout=20000)

    results = []
    result_blocks = await page.query_selector_all("li.ResultItem")
    for block in result_blocks:
        title_element = await block.query_selector("h3 a span")
        link_element = await block.query_selector("h3 a")
        snippet_element = await block.query_selector("div.abstract-preview-text")

        if title_element and link_element:
            title = await title_element.inner_text()
            url = await link_element.get_attribute("href")
            if url and not url.startswith("http"):
                url = "https://www.sciencedirect.com" + url
            snippet = await snippet_element.inner_text() if snippet_element else "No snippet available."
            results.append({"title": title, "url": url, "snippet": snippet})
    return results

async def _scrape_springer(page: Page, query: str) -> List[Dict[str, Any]]:
    """Scraper for SpringerLink."""
    search_url = f"https://link.springer.com/search?query={quote_plus(query)}"
    await page.goto(search_url, wait_until="domcontentloaded", timeout=20000)
    await page.wait_for_selector("li.app-card-open", timeout=15000)

    results = []
    result_blocks = await page.query_selector_all("li.app-card-open")
    for block in result_blocks:
        title_element = await block.query_selector("h3 a")
        snippet_element = await block.query_selector("p.app-card-body__snippet")
        
        if title_element:
            title = await title_element.inner_text()
            url = await title_element.get_attribute("href")
            if url and not url.startswith("http"):
                url = "https://link.springer.com" + url
            snippet = await snippet_element.inner_text() if snippet_element else "No snippet available."
            results.append({"title": title, "url": url, "snippet": snippet.strip()})
    return results

async def _scrape_ieee(page: Page, query: str) -> List[Dict[str, Any]]:
    """Scraper for IEEE Xplore."""
    search_url = f"https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText={quote_plus(query)}"
    await page.goto(search_url, wait_until="networkidle", timeout=30000) # IEEE is slow and heavy on JS
    
    # Handle cookie/privacy pop-ups if they exist
    if await page.query_selector("#onetrust-accept-btn-handler"):
        await page.click("#onetrust-accept-btn-handler")
        await page.wait_for_timeout(1000) # wait for popup to disappear

    await page.wait_for_selector("div.List-results-items", timeout=20000)
    
    results = []
    result_blocks = await page.query_selector_all("div.List-results-items")
    for block in result_blocks:
        title_element = await block.query_selector("h3 a")
        snippet_element = await block.query_selector("div.abstract-text")
        
        if title_element:
            title = await title_element.inner_text()
            url = await title_element.get_attribute("href")
            if url and not url.startswith("http"):
                url = "https://ieeexplore.ieee.org" + url
            snippet = await snippet_element.inner_text() if snippet_element else "No snippet available."
            results.append({"title": title, "url": url, "snippet": snippet.strip()})
    return results

async def _scrape_nasa_gov(page: Page, query: str) -> List[Dict[str, Any]]:
    """Scraper specifically for nasa.gov search."""
    search_url = f"https://www.nasa.gov/search/?query={quote_plus(query)}"
    await page.goto(search_url, wait_until="domcontentloaded", timeout=20000)
    await page.wait_for_selector("#search-results", timeout=15000)
    
    results = []
    # NASA search results are within this container
    result_blocks = await page.query_selector_all("div.list-item")
    for block in result_blocks:
        title_element = await block.query_selector("h3 a")
        snippet_element = await block.query_selector("p.preview-text")
        
        if title_element:
            title = await title_element.inner_text()
            url = await title_element.get_attribute("href")
            snippet = await snippet_element.inner_text() if snippet_element else "No abstract available."

            results.append({"title": title, "url": url, "snippet": snippet})
    return results

async def _scrape_esa(page: Page, query: str) -> List[Dict[str, Any]]:
    """Scraper for the European Space Agency (ESA)."""
    search_url = f"https://www.esa.int/esearch?q={quote_plus(query)}"
    await page.goto(search_url, wait_until="domcontentloaded", timeout=20000)

    results = []
    result_blocks = await page.query_selector_all("a.heading")
    for block in result_blocks:
        title = await block.inner_text()
        url = await block.get_attribute("href")
        
        # ESA search doesn't provide snippets on the results page
        results.append({"title": title.strip(), "url": url, "snippet": "No snippet available on ESA search results page."})
    return results

async def _scrape_duckduckgo(page: Page, query: str) -> List[Dict[str, Any]]:
    """Scraper for DuckDuckGo."""
    search_url = f"https://duckduckgo.com/?q={quote_plus(query)}&ia=web"
    await page.goto(search_url, wait_until="domcontentloaded", timeout=20000)
    await check_for_captcha(page)
    
    results = []
    result_blocks = await page.query_selector_all("article[data-testid='result']")
    for block in result_blocks:
        title_el = await block.query_selector("a[data-testid='result-title-a']")
        snippet_el = await block.query_selector("div[data-testid='result-snippet']")
        if title_el:
            title = await title_el.inner_text()
            url = await title_el.get_attribute("href")
            snippet = await snippet_el.inner_text() if snippet_el else "No snippet."
            results.append({"title": title, "url": url, "snippet": snippet})
    return results

async def _scrape_google(page: Page, query: str) -> List[Dict[str, Any]]:
    """Scraper for Google Search."""
    search_url = f"https://www.google.com/search?q={quote_plus(query)}"
    await page.goto(search_url, wait_until="domcontentloaded", timeout=20000)
    if not await check_for_captcha(page): return []
    
    results = []
    result_blocks = await page.query_selector_all("div.g, div.kvH3mc")
    for block in result_blocks:
        h3 = await block.query_selector("h3")
        a = await block.query_selector("a")
        if h3 and a:
            url = await a.get_attribute("href")
            if url and "google.com" not in url:
                results.append({"title": await h3.inner_text(), "url": url, "snippet": await block.inner_text()})
    return results

async def _scrape_searx(page: Page, query: str) -> List[Dict[str, Any]]:
    """Scraper for Searx (using a public instance)."""
    search_url = f"https://search.inetol.net/search?q={quote_plus(query)}"
    await page.goto(search_url, wait_until="domcontentloaded", timeout=20000)
    
    results = []
    result_blocks = await page.query_selector_all('div.result, article.result')
    for block in result_blocks:
        a = await block.query_selector("a")
        if a:
            url = await a.get_attribute("href")
            results.append({"title": await a.inner_text(), "url": url, "snippet": (await block.inner_text())[:200]})
    return results

async def _scrape_base(page: Page, query: str) -> List[Dict[str, Any]]:
    """Scraper for BASE (Bielefeld Academic Search Engine)."""
    search_url = f"https://www.base-search.net/Search/Results?lookfor={quote_plus(query)}"
    await page.goto(search_url, wait_until="domcontentloaded", timeout=20000)
    
    results = []
    result_blocks = await page.query_selector_all('div.record')
    for block in result_blocks:
        a = await block.query_selector('a.title')
        if a:
            results.append({"title": await a.inner_text(), "url": await a.get_attribute("href"), "snippet": (await block.inner_text())[:200]})
    return results

async def _scrape_ncbi(page: Page, query: str) -> List[Dict[str, Any]]:
    """Scraper for NCBI PubMed."""
    search_url = f"https://pubmed.ncbi.nlm.nih.gov/?term={quote_plus(query)}"
    await page.goto(search_url, wait_until="domcontentloaded", timeout=20000)
    
    results = []
    result_blocks = await page.query_selector_all('article.full-docsum')
    for block in result_blocks:
        a = await block.query_selector('a.docsum-title')
        if a:
            results.append({"title": await a.inner_text(), "url": await a.get_attribute("href"), "snippet": (await block.inner_text())[:200]})
    return results

async def _scrape_mdpi(page: Page, query: str) -> List[Dict[str, Any]]:
    """Scraper for MDPI."""
    search_url = f"https://www.mdpi.com/search?q={quote_plus(query)}"
    await page.goto(search_url, wait_until="domcontentloaded", timeout=20000)
    
    results = []
    result_blocks = await page.query_selector_all('article.article-item')
    for block in result_blocks:
        a = await block.query_selector('a.title-link')
        if a:
            results.append({"title": await a.inner_text(), "url": await a.get_attribute("href"), "snippet": (await block.inner_text())[:200]})
    return results

async def _scrape_tandf(page: Page, query: str) -> List[Dict[str, Any]]:
    """Scraper for Taylor & Francis."""
    search_url = f"https://www.tandfonline.com/action/doSearch?AllField={quote_plus(query)}"
    await page.goto(search_url, wait_until="domcontentloaded", timeout=25000)
    
    results = []
    result_blocks = await page.query_selector_all('div.searchResultItem')
    for block in result_blocks:
        a = await block.query_selector('a.hlFld-Title')
        if a:
            results.append({"title": await a.inner_text(), "url": await a.get_attribute("href"), "snippet": (await block.inner_text())[:200]})
    return results

async def _scrape_aiaa(page: Page, query: str) -> List[Dict[str, Any]]:
    """Scraper for AIAA."""
    search_url = f"https://arc.aiaa.org/action/doSearch?AllField={quote_plus(query)}"
    await page.goto(search_url, wait_until="domcontentloaded", timeout=25000)
    
    results = []
    result_blocks = await page.query_selector_all('div.search-item')
    for block in result_blocks:
        a = await block.query_selector('h3.item__title a')
        if a:
            results.append({"title": await a.inner_text(), "url": await a.get_attribute("href"), "snippet": (await block.inner_text())[:200]})
    return results
    
# --- Placeholder/Difficult Scrapers ---
# These sites are either very difficult to scrape due to bot detection (AIAA, TandF),
# or require specific APIs/complex navigation.
# A full implementation would be significantly more involved.

async def _scrape_placeholder(page: Page, query: str, engine_name: str) -> List[Dict[str, Any]]:
    logger.warning(f"Scraper for '{engine_name}' is a placeholder and not implemented.")
    return []

# --- Orchestrator ---

ENGINE_MAP: Dict[str, Callable] = {
    # Core Web
    "google": _scrape_google,
    "ddg": _scrape_duckduckgo,
    "searx": _scrape_searx,
    # Core Academic
    "google_scholar": _scrape_google_scholar,
    "semantic_scholar": _scrape_semantic_scholar,
    "core": _scrape_core,
    "science_gov": _scrape_sciencegov,
    "baidu_scholar": _scrape_baidu_scholar,
    "refseek": _scrape_refseek,
    "base": _scrape_base,
    "ncbi": _scrape_ncbi,
    # Publishers
    "sciencedirect": _scrape_sciencedirect,
    "springer": _scrape_springer,
    "ieee": _scrape_ieee,
    "mdpi": _scrape_mdpi,
    "tandf": _scrape_tandf,
    "aiaa": _scrape_aiaa,
    # Space Agencies
    "nasa": _scrape_nasa_gov,
    "esa": _scrape_esa,
}

async def search_and_scrape_web_async(
    query: str, 
    engines: List[str],
    headless: bool = True,
    capture_visuals: bool = False,
    semantic_rank: bool = True,
    embeddings_model: Any = None,
    priority: int = ELP0
) -> List[Dict[str, Any]]:
    """
    Performs web searches on a list of specified engines using Playwright
    and returns a combined, deduplicated list of results.
    
    Args:
        query (str): The search query.
        engines (List[str]): A list of engine keys from ENGINE_MAP to use.
        headless (bool): Whether to run the browser in headless mode.
    """
    if not PLAYWRIGHT_AVAILABLE:
        logger.error("Web search failed: Playwright library is not available.")
        return []

    all_results = []
    processed_urls = set()
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            java_script_enabled=True,
            ignore_https_errors=True,
        )
        
        tasks = []
        # Create a list of tasks to run concurrently
        for engine_name in engines:
            if engine_name not in ENGINE_MAP:
                logger.warning(f"No scraper found for engine: '{engine_name}'. Skipping.")
                continue
            tasks.append(
                _run_scraper_for_engine(context, query, engine_name)
            )

        # Run all scraping tasks in parallel
        engine_results_list = await asyncio.gather(*tasks)

        # Process results from all engines
        for engine_results in engine_results_list:
            if not engine_results: continue
            for result in engine_results:
                if result.get("url") and result["url"] not in processed_urls:
                    # Initialize result with basic fields
                    result["apa7_reference"] = generate_apa7_reference(result["title"], result["url"])
                    all_results.append(result)
                    processed_urls.add(result["url"])
        
        # Deep inspection/Visual capture (sequential to avoid being blocked by too many parallel visits)
        if capture_visuals and all_results:
            logger.info(f"Starting deep inspection of {len(all_results)} results...")
            inspector_page = await context.new_page()
            for r in all_results[:10]: # Limit to top 10 for performance
                try:
                    await inspector_page.goto(r["url"], wait_until="domcontentloaded", timeout=15000)
                    await asyncio.sleep(1)
                    r["screenshot_base64"] = await capture_base64_screenshot(inspector_page)
                    r["web_images"] = await extract_web_images(inspector_page)
                    # Update snippet if the search engine one was poor
                    page_text = await inspector_page.inner_text("body")
                    if len(r.get("snippet", "")) < 50:
                        r["snippet"] = page_text[:400].replace("\n", " ") + "..."
                except: continue
            await inspector_page.close()

        await browser.close()

    # Semantic Ranking
    if semantic_rank and all_results:
        logger.info(f"Ranking {len(all_results)} results semantically (Priority: ELP{priority})...")
        q_emb = get_embedding(query, embeddings_model=embeddings_model, priority=priority)
        if q_emb is not None:
            ranked = []
            for r in all_results:
                text_to_embed = f"{r['title']} {r.get('snippet', '')}"
                r_emb = get_embedding(text_to_embed, embeddings_model=embeddings_model, priority=priority)
                if r_emb is not None:
                    score = np.dot(q_emb, r_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(r_emb))
                    ranked.append((float(score), r))
                else:
                    ranked.append((0.0, r))
            ranked.sort(key=lambda x: x[0], reverse=True)
            all_results = []
            for i, (score, r) in enumerate(ranked):
                r["semantic_rank"] = i + 1
                r["semantic_score"] = score
                all_results.append(r)

    logger.info(f"Completed all web searches. Total unique results found: {len(all_results)}")
    return all_results

async def _run_scraper_for_engine(
    context: BrowserContext,
    query: str,
    engine_name: str
) -> Optional[List[Dict[str, Any]]]:
    """
    Worker function to run a single scraper in its own page context.
    This isolates failures and allows for parallel execution.
    """
    page = await context.new_page()
    scraper_func = ENGINE_MAP.get(engine_name)
    log_prefix = f"WebFetch|{engine_name}"
    
    try:
        logger.info(f"{log_prefix}: Starting search for query: '{query[:70]}...'")
        engine_results = await scraper_func(page, query)
        
        # Add source engine to each result
        for result in engine_results:
            result["source_engine"] = engine_name
            
        logger.success(f"{log_prefix}: Scrape successful. Found {len(engine_results)} results.")
        return engine_results

    except PlaywrightTimeoutError:
        logger.error(f"❌ {log_prefix}: Page timed out. The website might be slow or blocking requests.")
    except Exception as e:
        logger.error(f"❌ {log_prefix}: An unexpected error occurred during scrape: {e}")
        logger.exception(f"{log_prefix} Traceback:")
    finally:
        await page.close()
    
    return None


# --- Example Usage ---

async def main():
    """Main function to demonstrate the scraper."""
    # query = "black hole information paradox"
    query = "applications of generative adversarial networks in astrophysics"
    
    # Select a few engines to run
    engines_to_use = [
        "google",
        "ddg",
        "google_scholar",
        "nasa"
    ]
    
    logger.info(f"Starting async web scrape for query: '{query}' on engines: {engines_to_use}")
    
    # Set headless=False to watch the browser work (if needed)
    results = await search_and_scrape_web_async(
        query, 
        engines_to_use, 
        headless=True, 
        capture_visuals=True, 
        semantic_rank=True
    )
    
    print("\n--- SCRAPE RESULTS ---")
    if results:
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Title: {result.get('title')}")
            print(f"   URL: {result.get('url')}")
            print(f"   Source: {result.get('source_engine')}")
            if 'semantic_rank' in result:
                print(f"   Semantic Rank: {result['semantic_rank']} (Score: {result['semantic_score']:.4f})")
            snippet = result.get('snippet', 'N/A').replace('\n', ' ').strip()
            print(f"   Snippet: {snippet[:200]}...")
            if result.get('screenshot_base64'):
                print(f"   [Screenshot captured]")
            if result.get('web_images'):
                print(f"   [Images extracted: {len(result['web_images'])}]")
    else:
        print("No results found.")
    print("\n--- END OF RESULTS ---\n")


if __name__ == "__main__":
    if not PLAYWRIGHT_AVAILABLE:
        exit(1)
        
    # Configure Loguru for better output
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )

    asyncio.run(main())