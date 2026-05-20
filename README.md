# CLI Multi-Search Engine (Selenium Version)

A command-line tool to query multiple general and academic search engines simultaneously using Selenium with a visible browser, allowing for manual CAPTCHA intervention and optional content downloading.

**Disclaimer:** Web searching is inherently fragile and may break if the target websites change their structure. Google, Google Scholar, and publisher sites like ScienceDirect, MDPI, Taylor & Francis, IEEE Xplore, and SpringerLink are particularly difficult to search reliably. Performance with Selenium is generally slower than direct HTTP requests. Use responsibly and respect website Terms of Service.

## Features

*   Fetches results via Selenium (visible Chrome browser) from:
    *   General: DuckDuckGo (`ddg`), Google (`google`), SearX (`searx`)
    *   Academic: Google Scholar (`scholar`), BASE (`base`), CORE (`core`), Science.gov (`scigov`), Semantic Scholar (`sem`), Baidu Scholar (`baidu`), RefSeek (`refseek`), ScienceDirect (`scidirect`), MDPI (`mdpi`), Taylor & Francis (`tandf`), IEEE Xplore (`ieee`), SpringerLink (`springer`)
*   **Manual CAPTCHA Handling:** Pauses execution and prompts user in the terminal if a likely CAPTCHA is detected, allowing manual solving in the browser.
*   **Optional Content Downloading:** Can download the HTML content of result URLs and detected PDF links (`--download`, `--download-dir`). Note: Paywalled content will likely not download successfully.
*   Provides URLs, snippets, and sometimes direct PDF links extracted from result pages.
*   Executable script format (`./search_cli`).
*   Configurable SearX instances (via `config.json` or command-line).
*   Configurable timeout (`-t`).
*   Configurable number of results per engine (`-n`).
*   Configurable maximum pages to search (`--max-pages`, currently supports DuckDuckGo, Google, and Google Scholar).
*   Optional JSON (`--json`) or CSV (`--csv`) output.
*   Result deduplication based on URL (`--dedup-mode url`, default) or title similarity (`--dedup-mode title`).
*   Configurable similarity threshold (`--similarity-threshold`, default 0.8) for title deduplication.
*   Optional disabling of image loading (`--no-images`) for potentially faster searching.
*   Improved robustness for Google searching (multiple selectors, raw link fallback).

## Setup

1.  Ensure Python 3 and Google Chrome browser are installed.
2.  Create a virtual environment: `python3 -m venv venv`
3.  Activate the environment: `source venv/bin/activate`
4.  Install dependencies: `pip install selenium webdriver-manager requests beautifulsoup4 lxml`
5.  Make the script executable: `chmod +x search_cli`

## Usage

```bash
# Activate virtual environment first: source venv/bin/activate

# Basic search using default engines
./search_cli "your search query"

# Search specific engines, including SpringerLink
./search_cli "computational fluid dynamics" --engines scholar ieee springer

# Search all available engines (now includes all added publishers)
./search_cli "latest AI research" --engines all

# Specify number of results per engine
./search_cli "climate change impact" -n 5

# Specify timeout (in seconds) for navigation/waits
./search_cli "complex query" -t 25 # Increase timeout for publisher sites

# Search multiple pages (e.g., 3 pages for DDG, Google, Scholar)
./search_cli "web searching best practices" --engines ddg google scholar --max-pages 3

# Disable image loading for potentially faster searching
./search_cli "fast search" --engines ddg --no-images

# Specify custom SearX instances (if using 'searx' engine)
./search_cli "privacy respecting search" --engines searx --searx-instances https://searx.example.org/ https://another.searx.instance/

# Combine options
./search_cli "biomedical engineering trends" --engines scholar core scidirect mdpi tandf ieee springer -n 10 -t 25

# Output results as JSON
./search_cli "bioinformatics tools" --engines base sem --json

# Output results as CSV (includes PDF_URL column if found)
./search_cli "renewable energy sources" --engines scholar scidirect springer --csv > results.csv

# Deduplicate based on title similarity (threshold 0.85)
./search_cli "benefits of exercise" --engines ddg google --dedup-mode title --similarity-threshold 0.85

# Search and download results (HTML/PDF) to default ./Downloads directory
./search_cli "python requests library" --engines ddg google --download

# Search and download results to a specific directory
./search_cli "machine learning applications" --engines scholar springer ieee --download --download-dir ./MyDownloads

# Deactivate environment when done: deactivate
```

## TODO / Potential Improvements

*   **Improve Searcher Reliability:** Selectors need constant monitoring and updates, especially for publisher sites. Test and refine selectors for SciDirect, MDPI, T&F, IEEE, Springer.
*   **Error Handling:** More granular error handling for Selenium exceptions and download errors.
*   **CAPTCHA Handling:** Improve detection accuracy. Explore alternative handling.
*   **Pagination:** Implement robust pagination for publisher engines (SciDirect, MDPI, T&F, IEEE, Springer) and others (SearX, BASE, CORE, etc.).
*   **PDF Extraction:** Improve PDF link finding, potentially by navigating to article pages (more complex, slower, higher risk of blocks).
*   **Performance:** Explore further optimizations (e.g., parallel downloads).
*   **Configuration:** Move more options to `config.json`.
*   **Browser Choice:** Add support for other browsers.
*   **Download Robustness:** Add retries, better error handling, content type sniffing improvements in `download_utils`.
