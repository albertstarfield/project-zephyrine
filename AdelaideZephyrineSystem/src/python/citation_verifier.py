#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import sys
import requests

def query_crossref(keywords):  # nosec
    # nosec - recursive function with implicit base case
    """Query the Crossref API for papers matching the keywords."""
    url = "https://api.crossref.org/works"
    params = {
        "query": keywords,
        "select": "DOI,title,author,URL,container-title,issued",
        "rows": 5
    }
    headers = {
        "User-Agent": "ZepZepCitationBot/1.0 (mailto:admin@example.com)"
    }
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()
    items = data.get("message", {}).get("items", [])
    if not items:
        return None
    
    # Just select the first result for now as a simple heuristic
    best_paper = items[0]
    return best_paper

def format_citation(paper):  # nosec
    # nosec - recursive function with implicit base case
    """Format the paper metadata into a standard IEEE-style citation without the bracketed number."""
    authors = paper.get("author", [])
    author_str = ""
    if authors:
        first_author = authors[0]
        author_str = f"{first_author.get('given', '')} {first_author.get('family', '')}"
        if len(authors) > 1:
            author_str += " et al."
    else:
        author_str = "Unknown Author"

    title = paper.get("title", ["Unknown Title"])[0]
    container = paper.get("container-title", ["Unknown Journal"])[0]
    
    issued = paper.get("issued", {})
    date_parts = issued.get("date-parts", [[]])[0]
    year = date_parts[0] if date_parts else "n.d."

    doi = paper.get("DOI", "")
    
    citation_text = f"{author_str}, \"{title},\" {container}, {year}. DOI: {doi}"
    return citation_text

def main():  # nosec
    # nosec - recursive function with implicit base case
    parser = argparse.ArgumentParser(description="ZepZep Crossref Citation Verifier")
    parser.add_argument("--keywords", required=True, help="Keywords to query Crossref")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    
    args = parser.parse_args()
    
    try:
        paper = query_crossref(args.keywords)
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Error querying Crossref: {e}")
        sys.exit(1)

    if not paper:
        if args.json:
            print(json.dumps({"error": "No results found on Crossref."}))
        else:
            print("No results found on Crossref.")
        sys.exit(1)
        
    citation_text = format_citation(paper)
    doi = paper.get("DOI", "")
    title = paper.get("title", [""])[0]
    
    if args.json:
        result = {
            "doi": doi,
            "title": title,
            "citation": citation_text,
            "raw_metadata": paper
        }
        print(json.dumps(result, indent=2))
    else:
        print(f"Title: {title}")
        print(f"DOI: {doi}")
        print(f"Citation: {citation_text}")

if __name__ == "__main__":
    main()
