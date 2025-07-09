# network_internet_knowledge_fetcher.py

import asyncio
from urllib.parse import quote_plus
from loguru import logger
from typing import Optional, Dict, Any, List, Callable

try:
    from playwright.async_api import async_playwright, Page, BrowserContext, TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    logger.critical("Playwright library not found. Please run 'pip install playwright' and 'playwright install'.")
    PLAYWRIGHT_AVAILABLE = False

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
    
# --- Placeholder/Difficult Scrapers ---
# These sites are either very difficult to scrape due to bot detection (AIAA, TandF),
# or require specific APIs/complex navigation.
# A full implementation would be significantly more involved.

async def _scrape_placeholder(page: Page, query: str, engine_name: str) -> List[Dict[str, Any]]:
    logger.warning(f"Scraper for '{engine_name}' is a placeholder and not implemented.")
    return []

# --- Orchestrator ---

ENGINE_MAP: Dict[str, Callable] = {
    # Core Academic
    "google_scholar": _scrape_google_scholar,
    "semantic_scholar": _scrape_semantic_scholar,
    "core": _scrape_core,
    "science_gov": _scrape_sciencegov,
    "baidu_scholar": _scrape_baidu_scholar,
    "refseek": _scrape_refseek,
    # Publishers
    "sciencedirect": _scrape_sciencedirect,
    "springer": _scrape_springer,
    "ieee": _scrape_ieee,
    # Space Agencies
    "nasa": _scrape_nasa_gov,
    "esa": _scrape_esa,
    # Placeholders for difficult/unimplemented sites
    "mdpi": lambda p, q: _scrape_placeholder(p, q, "MDPI"),
    "tandf": lambda p, q: _scrape_placeholder(p, q, "Taylor & Francis"),
    "aiaa": lambda p, q: _scrape_placeholder(p, q, "AIAA"),
    "arc": lambda p, q: _scrape_placeholder(p, q, "NASA ARC (NTRS)"),
}

async def search_and_scrape_web_async(
    query: str, 
    engines: List[str],
    headless: bool = True
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
            if not engine_results:
                continue
            
            for result in engine_results:
                if result.get("url") and result["url"] not in processed_urls:
                    all_results.append(result)
                    processed_urls.add(result["url"])
        
        await browser.close()

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
    # To run all: engines_to_use = list(ENGINE_MAP.keys())
    engines_to_use = [
        "google_scholar",
        "semantic_scholar",
        "springer",
        "nasa",
        "esa"
    ]
    
    logger.info(f"Starting async web scrape for query: '{query}' on engines: {engines_to_use}")
    
    # Set headless=False to watch the browser work
    results = await search_and_scrape_web_async(query, engines_to_use, headless=True)
    
    print("\n--- SCRAPE RESULTS ---")
    if results:
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Title: {result.get('title')}")
            print(f"   URL: {result.get('url')}")
            print(f"   Source: {result.get('source_engine')}")
            snippet = result.get('snippet', 'N/A').replace('\n', ' ').strip()
            print(f"   Snippet: {snippet[:200]}...")
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