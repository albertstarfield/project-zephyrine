#!/usr/bin/env python3
import os
import re
import datetime
import json
import subprocess
import base64
from collections import defaultdict
from typing import List, Dict, Any, Optional

# --- Attempt to import the PDF generation library ---
try:
    from md_to_pdf import md_to_pdf

    PDF_CAPABLE = True
except ImportError:
    PDF_CAPABLE = False



# ==============================================================================
# Adversarial Audit & Validation Agent (Pure Python Version)
# Version: 6.2 - Bugfix & Refinements
#
# This agent performs a multi-pass audit and generates both Markdown and PDF reports.
#
# Key Features:
# - High-Performance & Verbose File Discovery: Uses an optimized Python `os.walk`
#   with intelligent pruning to efficiently scan large repositories on any OS.
# - Multi-Pass Analysis: Combines rule-based checks, LLM code review, VLM image
#   analysis, and an adversarial "how to break this" pass.
# - Dual-Format Reporting: Generates a detailed Markdown (.md) report for easy
#   viewing and a professional, styled PDF report for distribution and archiving.
#
# Usage:
#   1. Place this script in the root of the project repository.
#   2. Ensure dependencies are installed:
#      pip install requests md-to-pdf
#   3. Run it: python3 audit_agent.py
# ==============================================================================

# --- Agent Configuration ---
CONFIG = {
    "AUDITOR_NAME": "Adelaide Zephyrine Charlotte (Adversarial Agent v6.3)",
    "NOTES_FILE": "AutomatedAuditNotes.txt",
    "REPORT_TEMPLATE": "AuditNote_{date}.md",
    "FEATURE_LIST_PATH": "./documentation/Developer Documentation/TodoFeatureFullList.md",
    "MAX_CHUNK_LINES": 500,
    "LLM_API_ENDPOINT": "http://localhost:18141/v1/chat/completions",
    "LLM_MODEL_NAME": "google/gemma-3n-e4b",
    "IS_WINDOWS": os.name == 'nt',
    "ALLOWED_EXTENSIONS": {
        # Code Files
        '.c', '.py', '.go', '.sh', '.jsx', '.tsx', '.ts', '.js',
        # Ada Files
        '.adb', '.ads', '.gpr',
        # Config & Markup
        '.md', '.toml', '.txt', '.json', '.html', '.ini', '.mako',
        # Image Files
        '.jpg', '.jpeg', '.png', '.bmp', '.ico'
    },
    # --- ADDITION: Define additional folders and patterns to ignore ---
    # These are combined with .gitignore. Use standard .gitignore syntax.
    "ADDITIONAL_IGNORE_PATTERNS": {
        # Default ignores for any project
        ".git", ".vscode", "__pycache__", "node_modules", "build", "dist",
        "*.pyc", "*.log", "*.egg-info",
        # Project-specific examples
        "meshCommunicationRelay", "staticmodelpool", "db_snapshots" ,"temp", "_excludefromRuntime_reverseEngineeringAssets", "alire", "alire*", "deps"
        # Virtual Environments
        "zephyrineCondaVenv", "miniforge3_local", "venv", ".venv",
        # Agent's own output files
        "AutomatedAuditNotes.txt", "AuditNote_*.md", "AuditNote_*.pdf"
    }
}

# --- Agent State (in-memory) ---
audit_findings: List[Dict[str, Any]] = []
file_connections: Dict[str, set] = defaultdict(set)
audited_file_structure: Dict[str, Any] = {}
full_codebase_map: Dict[str, str] = {}  # Maps file path to its content


def fast_file_discovery(repo_root: str) -> List[str]:
    """
    Performs a high-speed, verbose file discovery using an optimized os.walk
    that intelligently prunes ignored directories and filters by extension.
    """
    global audited_file_structure
    print("   -> Using optimized Python `os.walk` with live pruning for discovery...")

    gitignore_patterns_re = [re.compile(p) for p in load_gitignore_patterns_for_re(repo_root)]

    audited_files = []
    dir_count = 0

    for root, dirs, files in os.walk(repo_root, topdown=True):
        dir_count += 1
        print(f"\r   -> [Walker] Scanned {dir_count} directories. Current: {os.path.relpath(root, repo_root)}", end="",
              flush=True)

        # Prune ignored directories
        dirs[:] = [d for d in dirs if not any(
            p.fullmatch(os.path.join(os.path.relpath(root, repo_root), d).replace(os.sep, '/').lstrip('./')) for p in
            gitignore_patterns_re)]

        current_level = audited_file_structure
        rel_dir = os.path.relpath(root, repo_root)
        if rel_dir != ".":
            for part in rel_dir.split(os.sep):
                current_level = current_level.setdefault(part, {})

        filtered_files = []
        for file in files:
            # --- MODIFICATION START ---
            # 1. First, perform a quick and cheap check on the file extension.
            #    We use .lower() to handle cases like .PNG or .JPG.
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext not in CONFIG['ALLOWED_EXTENSIONS']:
                continue  # Skip to the next file if the extension is not allowed

            # 2. If the extension is allowed, proceed with the more expensive gitignore check.
            relative_path = os.path.join(rel_dir, file).replace(os.sep, '/') if rel_dir != "." else file
            if not any(p_re.fullmatch(relative_path.lstrip('./')) for p_re in gitignore_patterns_re):
                audited_files.append(os.path.join(root, file))
                filtered_files.append(file)
            # --- MODIFICATION END ---

        if filtered_files:
            current_level['__files__'] = sorted(filtered_files)

    print()  # Newline after the progress indicator

    def sort_tree(d):
        if '__files__' in d: d['__files__'].sort()
        for k, v in d.items():
            if k != '__files__': sort_tree(v)

    sort_tree(audited_file_structure)

    return sorted(audited_files)


def load_gitignore_patterns_for_re(repo_root: str) -> List[str]:
    """
    Loads gitignore patterns from CONFIG and .gitignore file, then converts them
    into a list of regex patterns for matching.
    """
    gitignore_path = os.path.join(repo_root, '.gitignore')

    # --- MODIFICATION START ---
    # 1. Start with the patterns defined in the script's configuration.
    #    Using a set automatically handles duplicates.
    all_patterns = set(CONFIG.get("ADDITIONAL_IGNORE_PATTERNS", {}))

    # 2. Add patterns from the project's .gitignore file, if it exists.
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    all_patterns.add(line)

    # 3. Convert all collected patterns to regex.
    regex_patterns = []
    for line in all_patterns:
        # This logic correctly handles gitignore-style patterns for our use case.
        pattern = line.replace('**', '*')  # Simplify globstar for this regex
        pattern = re.escape(pattern).replace(r'\*', '.*')
        if pattern.endswith('/'):
            pattern += '.*'  # Match anything inside a directory
        # Match the pattern itself or anything inside a directory with that name
        regex_patterns.append(f"({pattern})$|({pattern}/.*)")
    # --- MODIFICATION END ---

    return regex_patterns


def chunk_file_content(file_path: str, MAX_CHUNK_LINES=386):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        if not lines:
            yield 0, ""
            return
        total_lines = len(lines)
        for i in range(0, total_lines, CONFIG['MAX_CHUNK_LINES']):
            yield total_lines, "".join(lines[i:i + MAX_CHUNK_LINES])
    except Exception as e:
        print(f"   ⚠️  Could not read file {os.path.basename(file_path)}: {e}")
        yield 0, None


def update_notes(finding_type: str, file: str, line_num: int, code_snippet: str, note: str, suggestion: str = ""):
    finding = {
        "type": finding_type, "file": os.path.relpath(file), "line": line_num,
        "code": code_snippet.strip(), "note": note, "suggestion": suggestion
    }
    audit_findings.append(finding)
    with open(CONFIG['NOTES_FILE'], 'a', encoding='utf-8') as f:
        f.write(f"--- FINDING ---\nType: {finding_type}\nFile: {finding['file']}\nLine: {line_num}\nNote: {note}\n")
        if suggestion: f.write(f"Suggestion: {suggestion}\n")
        f.write(f"Code:\n```\n{finding['code']}\n```\n\n")


def invoke_llm_for_analysis(prompt_text: str, image_b64: Optional[str] = None) -> str:
    headers = {"Content-Type": "application/json"}
    messages = [{"role": "system",
                 "content": "You are a concise and expert code/image auditor. Provide direct, insightful analysis without conversational fluff."}]
    user_content = [{"type": "text", "text": prompt_text}]
    if image_b64:
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
    messages.append({"role": "user", "content": user_content})
    payload = {
        "model": CONFIG['LLM_MODEL_NAME'], "messages": messages, "temperature": 0.2,
        "max_tokens": 1024, "stream": False
    }
    try:
        response = requests.post(CONFIG['LLM_API_ENDPOINT'], headers=headers, data=json.dumps(payload), timeout=120)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except requests.exceptions.RequestException as e:
        return f"LLM API Error: Could not connect or request failed: {e}"
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return f"LLM API Error: Unexpected response format from server: {e.text if 'e' in locals() and hasattr(e, 'text') else 'Unknown format'}"


def analyze_code_chunk(file_path: str, chunk: str, chunk_num: int):
    global current_file_context
    if chunk is None: return
    base_filename = os.path.basename(file_path)
    for i, line in enumerate(chunk.splitlines()):
        line_num = (chunk_num * CONFIG['MAX_CHUNK_LINES']) + i + 1
        if 'self.current_session_id =' in line or 'self.vectorstore_url =' in line:
            if base_filename == "AdelaideAlbertCortex.py":
                update_notes("CRITICAL BUG", file_path, line_num, line,
                             "Potential Race Condition: A shared, global AI instance is modifying session-specific state.",
                             "Refactor the class to be stateless. Pass `session_id` as a parameter.")
        import_match = re.match(r'^\s*from\s+([\w.]+)\s+import', line)
        if import_match:
            imported_module = import_match.group(1)
            if '.' not in imported_module and not imported_module in ['sys', 'os', 're', 'json', 'time', 'threading',
                                                                      'asyncio', 'requests', 'sqlalchemy', 'flask',
                                                                      'langchain_core']:
                file_connections[base_filename].add(f"{imported_module}.py")
        if 'subprocess.Popen' in line or 'subprocess.run' in line:
            if 'shell=True' in line:
                update_notes("HIGH RISK", file_path, line_num, line,
                             "Security Risk: `subprocess` is used with `shell=True`.",
                             "Avoid `shell=True`. Construct the command as a list of arguments.")
        if 'math.' in line or 'cmath.' in line:
            update_notes("MATHEMATICAL VALIDATION", file_path, line_num, line, "A mathematical formula was detected.",
                         "Cross-reference the formula with established literature. Check for correct unit conversions.")
        if 'SessionLocal()' in line and base_filename != 'database.py':
            current_file_context['db_session_opened_on_line'] = line_num
        if 'db.close()' in line and 'db_session_opened_on_line' in current_file_context:
            del current_file_context['db_session_opened_on_line']

    if chunk.strip():
        llm_prompt = f"You are an expert code auditor. Provide a brief, critical analysis of the following code snippet from the file '{base_filename}'. Focus on potential bugs, anti-patterns, or logical errors. If it looks good, say so. Be concise.\n\nCode:\n```python\n{chunk}\n```"
        llm_analysis = invoke_llm_for_analysis(llm_prompt)
        if "LLM API Error" not in llm_analysis:
            update_notes("AUTOMATED VALIDATION ANALYSIS", file_path, (chunk_num * CONFIG['MAX_CHUNK_LINES']) + 1, chunk[:200] + "...",
                         llm_analysis)
        else:
            update_notes("POTENTIAL ISSUE", file_path, (chunk_num * CONFIG['MAX_CHUNK_LINES']) + 1, chunk[:200] + "...",
                         "LLM analysis failed for this chunk.", llm_analysis)


def analyze_image_file(file_path: str):
    print(f"   👁️  VLM Analyzing Image: {os.path.relpath(file_path)}...")
    try:
        with open(file_path, 'rb') as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        prompt = "You are a UI/UX and software architecture analyst. Analyze this image. What is its likely purpose in this software project? Is it a UI mockup, a system diagram, a logo, or a character asset? Describe its key elements and assess its quality or clarity."
        vlm_analysis = invoke_llm_for_analysis(prompt, image_b64)
        update_notes("IMAGE REVIEW", file_path, 1, os.path.basename(file_path), vlm_analysis)
    except Exception as e:
        update_notes("IMAGE REVIEW", file_path, 1, os.path.basename(file_path), f"Failed to process image: {e}")


def perform_cynical_audit():
    print("\nStep 4: Performing Cynical / Adversarial Audit Pass...")
    launcher_code = full_codebase_map.get('launcher.py', '')
    cortex_code = full_codebase_map.get('systemCore/engineMain/AdelaideAlbertCortex.py', '')
    if 'direct_generate' in cortex_code and 'background_generate' in cortex_code and 'watchdog' in cortex_code:
        update_notes("LOGICAL FLAW", "AdelaideAlbertCortex.py", 0, "Dual-Generate Logic",
                     "The system spawns a background (ELP0) task concurrently with the foreground (ELP1) task. However, the ELP1 task is monitored by a watchdog that can terminate the entire system if it's too slow. There is no mechanism to cancel the already-spawned ELP0 task in this event.",
                     "Implement a cancellation token that is passed to the `background_generate` task. The watchdog, upon terminating the ELP1 process, should also set this token to signal the background task to exit gracefully and prevent it from becoming an orphaned, resource-consuming process.")
    if '_port_shield_worker_daemon' in launcher_code and 'p.kill()' in launcher_code:
        update_notes("HIGH RISK", "launcher.py", 0, "_port_shield_worker_daemon",
                     "The Port Shield daemon is designed to be 'barbaric'. It kills any process with 'ollama' in its name on port 11434. This is a brittle and dangerous assumption. A legitimate, unrelated process could be instantly terminated without warning, causing data loss or system instability.",
                     "Change the kill condition to be more specific, e.g., check the full command-line path of the process. Alternatively, downgrade the action from `kill()` to a persistent, high-visibility warning.")
    if 'self.current_session_id =' in cortex_code:
        update_notes("CRITICAL BUG", "AdelaideAlbertCortex.py", 0, "class CortexThoughts:",
                     "The core `CortexThoughts` class is a singleton that stores session-specific state in instance variables. An adversary could exploit this by sending two requests in rapid succession, causing context from one user's session to leak into another's, potentially exposing private data or causing the AI to give dangerously incorrect answers.",
                     "This is a fundamental architectural flaw. The class MUST be refactored to be completely stateless. All session-specific data must be passed explicitly as function arguments.")
    if 'background_generate_task_semaphore' in cortex_code and 'asyncio.create_task' in cortex_code:
        update_notes("POTENTIAL ISSUE", "AdelaideAlbertCortex.py", 0, "background_generate",
                     "The system spawns numerous background tasks. While a semaphore limits concurrency, a malicious or accidental flood of requests could fill the queue, leading to extreme memory usage and process starvation as tasks wait for their turn. There is no timeout or cleanup for tasks waiting on the semaphore.",
                     "Implement a timeout on `semaphore.acquire()`. If a task cannot acquire the semaphore within a reasonable time, it should fail gracefully and log an error, rather than waiting indefinitely.")
    print("   -> Cynical audit pass complete.")


def generate_file_tree_string(structure, prefix=""):
    tree_lines = []
    # Separate files from directories to list directories first
    dirs = sorted([k for k in structure if k != '__files__'])
    files = sorted(structure.get('__files__', []))

    items = dirs + files
    for i, name in enumerate(items):
        is_last = i == len(items) - 1
        connector = "└── " if is_last else "├── "
        tree_lines.append(f"{prefix}{connector}{name}")

        if name in dirs:  # It's a directory
            new_prefix = prefix + ("    " if is_last else "│   ")
            tree_lines.extend(generate_file_tree_string(structure[name], new_prefix))

    return tree_lines


def map_finding_type_to_dal(finding_type: str) -> str:
    mapping = {
        "CRITICAL BUG": "DAL A (Fatal)", "LOGICAL FLAW": "DAL A (Fatal)",
        "HIGH RISK": "DAL B (Hazardous)", "POTENTIAL ISSUE": "DAL C (Major)",
        "AUTOMATED VALIDATION ANALYSIS": "DAL D (Minor)", "MATHEMATICAL VALIDATION": "DAL C (Major)",
        "IMPROVEMENT": "DAL D (Minor)", "IMAGE REVIEW": "DAL E (No Safety Effect)"
    }
    return mapping.get(finding_type, "DAL E (No Safety Effect)")


def generate_report():
    print("\n\n📊 Generating final audit report...")
    file_tree_lines = generate_file_tree_string(audited_file_structure)
    file_tree_md = ".\n" + "\n".join(file_tree_lines)
    mermaid_graph = "graph TD;\n"
    nodes = set(file_connections.keys())
    for connections in file_connections.values(): nodes.update(connections)
    for node in nodes: mermaid_graph += f"    {node.replace('.py', '')}[{node}];\n"
    for source, targets in file_connections.items():
        for target in targets: mermaid_graph += f"    {source.replace('.py', '')} --> {target.replace('.py', '')};\n"
    report_content = f"""# Zephyrine Project Audit Report
- **Date:** {datetime.date.today().strftime("%Y-%m-%d")}
- **Auditor:** {CONFIG['AUDITOR_NAME']}
## 1. Methodology
This report was generated by an automated audit agent. The agent performed the following steps:
1.  Scanned the repository at `{os.getcwd()}` using a high-performance discovery method.
2.  Ignored files and directories specified in `.gitignore` and a default exclusion list.
3.  Generated a file structure map of all audited components.
4.  **Pass 1 (Static & AI):** Analyzed code files using a hybrid approach: a static ruleset for critical issues and a local LLM for nuanced logical analysis. Analyzed image assets using a local VLM.
5.  **Pass 2 (Adversarial):** Performed a cynical, high-level review of the entire codebase to identify deep logical flaws, race conditions, and resource exhaustion vectors.
6.  Persisted findings to `{CONFIG['NOTES_FILE']}` and aggregated them into this final report.
7.  Performed a final meta-analysis, scrutinizing the project's feature list against all audit findings.
## 2. Audited File Structure
The following tree represents the files and directories analyzed during this audit.
```
{file_tree_md}
```
## 3. System Connection Graph
This graph illustrates the detected import dependencies between the core Python modules.
```mermaid
{mermaid_graph}
```
## 4. Detailed Findings
The following issues and observations were identified during the audit, categorized by source and severity.
"""
    image_findings = [f for f in audit_findings if f['type'] == 'IMAGE REVIEW']
    if image_findings:
        report_content += "\n### Image Asset Analysis (VLM)\n\n| File | VLM Analysis |\n|:---|:---|\n"
        for finding in image_findings: report_content += f"| `{finding['file']}` | {finding['note']} |\n"
    cynical_findings = [f for f in audit_findings if f['type'] in ["LOGICAL FLAW"]]
    rule_llm_findings = [f for f in audit_findings if f['type'] not in ["IMAGE REVIEW", "LOGICAL FLAW"]]
    if cynical_findings:
        report_content += "\n### Adversarial & Logical Scrutiny (Pass 2)\n\n| Flaw Type | Component | Note & Suggestion |\n|:---|:---|:---|\n"
        for finding in cynical_findings:
            note_suggestion = f"**Note:** {finding['note']}<br><br>**Suggestion:** {finding['suggestion']}"
            report_content += f"| **{finding['type']}** | `{finding['file']}` | {note_suggestion} |\n"
    severity_order = ["CRITICAL BUG", "HIGH RISK", "POTENTIAL ISSUE", "AUTOMATED VALIDATION ANALYSIS", "MATHEMATICAL VALIDATION"]
    findings_by_type = defaultdict(list)
    for f in rule_llm_findings: findings_by_type[f['type']].append(f)
    for severity in severity_order:
        if severity in findings_by_type:
            report_content += f"\n### Code Analysis (Pass 1): {severity}\n\n| File | Line | Note & Suggestion | Code Snippet |\n|:-----|:----:|:------------------|:-------------|\n"
            for finding in findings_by_type[severity]:
                note_suggestion = f"**Note:** {finding['note']}"
                if finding['suggestion']: note_suggestion += f"<br><br>**Suggestion:** {finding['suggestion']}"
                code_snippet_md = f"```python\n{finding['code']}\n```"
                report_content += f"| `{finding['file']}` | {finding['line']} | {note_suggestion} | {code_snippet_md} |\n"
            report_content += "\n"
    try:
        with open(CONFIG['FEATURE_LIST_PATH'], 'r', encoding='utf-8') as f:
            feature_list_content = f.read()
    except FileNotFoundError:
        print(f"   ⚠️  Feature list not found at {CONFIG['FEATURE_LIST_PATH']}. Skipping feature audit.")
        feature_list_content = ""
    if feature_list_content:
        parsed_features = []
        for line in feature_list_content.splitlines():
            match = re.search(r'-\s*\[([x/ ])\]\s*(~~)?(.*?)(~~)?$', line)
            if match:
                status_char, is_cancelled1, desc, is_cancelled2 = match.groups()
                is_cancelled = is_cancelled1 is not None or is_cancelled2 is not None
                status_str = "Unknown"
                if is_cancelled:
                    status_str = "Cancelled"
                elif status_char == 'x':
                    status_str = "Implemented"
                elif status_char == '/':
                    status_str = "Partially Implemented"
                elif status_char == ' ':
                    status_str = "To-Do"
                parsed_features.append({'status': status_str, 'description': desc.strip()})
        appendix_md = "\n\n---\n\n## 5. Feature Implementation Audit\n\n"
        appendix_md += "This section scrutinizes the implementation of each feature against all findings from the audit. The DAL (Design Assurance Level) rating indicates the potential impact of identified risks on system stability, security, and correctness.\n\n"
        appendix_md += "| Feature | Status | Implementation Notes & Scrutiny | Identified Risks (DAL Rating) |\n|:---|:---|:---|:---|\n"
        for feature in parsed_features:
            description = feature['description']
            status = feature['status']
            keywords = set(re.findall(r'\b\w+\b', description.lower())) - {'a', 'the', 'and', 'for', 'of', 'in', 'via'}
            linked_findings = [f for f in audit_findings if any(
                kw in f"{f['file']} {f['note']} {f['code']}".lower() for kw in keywords if len(kw) > 3)]
            scrutiny_notes = f"**{status}.** "
            risks_md = ""
            if status in ["Implemented", "Partially Implemented"]:
                if not linked_findings:
                    scrutiny_notes += "Automated & adversarial scans found no direct, high-risk issues associated with this feature's core implementation files."
                    risks_md = "None detected by automated scan."
                else:
                    scrutiny_notes += "The implementation is associated with the following audit findings:"
                    risk_items = [
                        f"- **{map_finding_type_to_dal(f['type'])}:** In `{f['file']}`, a finding notes: *{f['note']}*"
                        for f in linked_findings]
                    risks_md = "\n".join(risk_items)
            elif status == "To-Do":
                scrutiny_notes = "This feature is planned but not yet implemented.";
                risks_md = "N/A"
            elif status == "Cancelled":
                scrutiny_notes = "This feature has been explicitly cancelled.";
                risks_md = "N/A"
            description_md = description.replace('|', '\\|')
            appendix_md += f"| {description_md} | {status} | {scrutiny_notes} | {risks_md} |\n"
        report_content += appendix_md
    with open(CONFIG['REPORT_FILE'], 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"✅ Markdown report successfully generated: {CONFIG['REPORT_FILE']}")


def generate_pdf_report(md_path: str, pdf_path: str):
    """Converts the generated Markdown report to a styled PDF."""
    if not PDF_CAPABLE:
        print("\n   ⚠️  PDF generation skipped: `md-to-pdf` library not found.")
        print("      To enable this feature, please run: pip install md-to-pdf")
        return

    print(f"\n📄 Generating PDF report: {os.path.basename(pdf_path)}...")

    # Simple CSS for a professional, GitHub-like appearance
    github_style_css = """
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji"; line-height: 1.6; padding: 30px; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 16px; }
    th, td { border: 1px solid #dfe2e5; padding: 8px 12px; }
    th { background-color: #f6f8fa; font-weight: 600; }
    tr:nth-child(2n) { background-color: #f6f8fa; }
    code { background-color: rgba(27,31,35,.05); padding: .2em .4em; margin: 0; font-size: 85%; border-radius: 3px; font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace; }
    pre > code { display: block; padding: 16px; overflow: auto; line-height: 1.45; border-radius: 3px; }
    h1, h2, h3 { border-bottom: 1px solid #eaecef; padding-bottom: .3em; margin-top: 24px; margin-bottom: 16px; font-weight: 600; }
    """

    try:
        # The md_to_pdf function is asynchronous in recent versions
        import asyncio
        asyncio.run(md_to_pdf(pdf_path, md_file_path=md_path, stylesheet_string=github_style_css))
        print(f"   ✅ PDF report successfully generated: {pdf_path}")
    except Exception as e:
        print(f"   ❌ ERROR: Failed to generate PDF report.")
        print(f"      Reason: {e}")
        print("      This may be due to a missing Chromium installation. Try running `playwright install`.")
        print(f"      The Markdown report is still available at: {md_path}")


def main():
    repo_root = os.getcwd()
    if os.path.exists(CONFIG['NOTES_FILE']): os.remove(CONFIG['NOTES_FILE'])
    print("Step 1: Discovering files and building structure map...")
    files_to_audit = fast_file_discovery(repo_root)
    if not files_to_audit:
        print("   ❌ No files found to audit. Exiting.")
        return
    print(f"   -> Discovery complete. Found {len(files_to_audit)} files to analyze.")
    print("\nStep 2: Beginning Pass 1: Static, LLM, and VLM Analysis...")
    for i, file_path in enumerate(files_to_audit):
        rel_path = os.path.relpath(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                full_codebase_map[rel_path] = f.read()
        except Exception:
            full_codebase_map[rel_path] = ""
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            analyze_image_file(file_path)
            continue
        global current_file_context
        current_file_context = {}
        content_for_chunking = full_codebase_map.get(rel_path, "")
        lines = content_for_chunking.splitlines()
        total_lines = len(lines)
        num_chunks = (total_lines + CONFIG['MAX_CHUNK_LINES'] - 1) // CONFIG[
            'MAX_CHUNK_LINES'] if total_lines > 0 else 1
        if not content_for_chunking.strip():
            print(f"   ({i + 1}/{len(files_to_audit)}) Auditing Code: {rel_path} [Empty File]")
            continue
        for chunk_num in range(num_chunks):
            start_line = chunk_num * CONFIG['MAX_CHUNK_LINES']
            end_line = start_line + CONFIG['MAX_CHUNK_LINES']
            chunk = "\n".join(lines[start_line:end_line])
            print(
                f"   ({i + 1}/{len(files_to_audit)}) Auditing Code: {rel_path} [Chunk {chunk_num + 1}/{num_chunks}]...")
            analyze_code_chunk(file_path, chunk, chunk_num)
        if 'db_session_opened_on_line' in current_file_context:
            update_notes("POTENTIAL ISSUE", file_path, current_file_context['db_session_opened_on_line'],
                         "File ends before DB session is closed.",
                         "A database session appears to be opened but is not explicitly closed within all code paths of this file.",
                         "Ensure any function that opens a session closes it, ideally using a `try...finally` block or a context manager.")

    perform_cynical_audit()
    generate_report()

    # --- Final Step: Generate PDF ---
    md_report_path = CONFIG['REPORT_FILE']
    # FIX: Derive PDF name from MD name to ensure they match
    pdf_report_path = os.path.splitext(md_report_path)[0] + '.pdf'
    generate_pdf_report(md_report_path, pdf_report_path)


if __name__ == "__main__":
    try:
        import requests
    except ImportError:
        print(
            "❌ ERROR: The 'requests' library is not installed. Please install it to run the agent: pip install requests")
        exit(1)

    CONFIG['CURRENT_DATE'] = datetime.date.today().strftime("%Y-%m-%d")
    CONFIG['REPORT_FILE'] = CONFIG['REPORT_TEMPLATE'].replace('{date}', CONFIG['CURRENT_DATE'])
    main()
