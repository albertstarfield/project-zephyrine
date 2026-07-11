import re
import urllib.request
import urllib.error
import sys

file_path = '/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/citations.bib'

with open(file_path, 'r') as f:
    lines = f.readlines()

cleaned_lines = []
removed_count = 0

print("Scanning for DOIs and verifying...")

for line in lines:
    match = re.search(r'doi\s*=\s*\{([^}]+)\}', line)
    if match:
        doi = match.group(1)
        url = f'https://doi.org/{doi}'
        valid = False
        try:
            req = urllib.request.Request(url, method='HEAD')
            with urllib.request.urlopen(req, timeout=5) as response:
                valid = True
        except:
            pass
        
        if valid:
            cleaned_lines.append(line)
        else:
            removed_count += 1
            print(f"Removed invalid DOI: {doi}")
    else:
        cleaned_lines.append(line)

with open(file_path, 'w') as f:
    f.writelines(cleaned_lines)

print(f"\nFinished. Removed {removed_count} invalid DOI fields from the file.")
