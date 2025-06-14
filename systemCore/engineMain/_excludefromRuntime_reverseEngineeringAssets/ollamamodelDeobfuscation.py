import subprocess
import json
import os
import shutil
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# --- Configuration ---
# ... (Keep configuration the same) ...
SCRIPT_DIR = Path(__file__).parent.resolve()
TARGET_GGUF_DIR = SCRIPT_DIR / "staticmodelpool"
OLLAMA_MODELS_DIR = Path.home() / ".ollama" / "models"
BLOB_DIR = OLLAMA_MODELS_DIR / "blobs"
MANIFEST_BASE_DIR = OLLAMA_MODELS_DIR / "manifests"

# ... (Keep OLLAMA_TO_LLAMA_CPP_FILENAME map the same) ...
OLLAMA_TO_LLAMA_CPP_FILENAME: Dict[str, str] = {
    "deepscaler:latest": "deepscaler.gguf",
    "gemma3:4b-it-qat": "gemma3--4b-it-qat.gguf", # Example VLM mapping
    "hf.co/mradermacher/LatexMind-2B-Codec-i1-GGUF:IQ4_XS": "hf--mradermacher--LatexMind-2B-Codec-i1-GGUF-IQ4_XS.gguf",
    "qwen2-math:1.5b-instruct-q5_K_M": "qwen2-math--1.5b-instruct-q5_K_M.gguf",
    "qwen2.5-coder:3b-instruct-q5_K_M": "qwen2.5-coder--3b-instruct-q5_K_M.gguf",
    "qwen3:0.6b-q4_K_M": "qwen3--0.6b-q4_K_M.gguf",
    "hf.co/mradermacher/NanoTranslator-immersive_translate-0.5B-GGUF:Q4_K_M": "hf.co--mradermacher--NanoTranslator-immersive_translate-0.5B-GGUF-Q4_K_M.gguf",
    "mxbai-embed-large:latest": "mxbai-embed-large-v1.gguf",
    "gemma3:1b-it-qat": "gemma3--1b-it-qat.gguf",
    "avil/UI-TARS:latest": "avil-ui-tars.gguf",
}


def run_command(command: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Runs a command and returns (stdout, stderr)."""
    # ... (Keep run_command the same) ...
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False, encoding='utf-8')
        if process.returncode != 0:
             print(f"[WARN] Command failed: {' '.join(command)}")
             print(f"       Stderr: {process.stderr.strip()}")
             return None, process.stderr.strip()
        return process.stdout.strip(), None
    except FileNotFoundError:
        print(f"[ERROR] Command not found: {command[0]}")
        return None, f"Command not found: {command[0]}"
    except Exception as e:
        print(f"[ERROR] Failed to run command {' '.join(command)}: {e}")
        return None, str(e)


# --- MODIFIED: More Robust find_manifest_file ---
def find_manifest_file(tag_full: str) -> Optional[Path]:
    """Attempts to find the manifest file path for a given Ollama tag."""
    print(f"  Attempting to find manifest for: {tag_full}")
    model_path_part = tag_full.rsplit(':', 1)[0]
    model_version_tag = tag_full.rsplit(':', 1)[1] # This is the actual filename

    print(f"    DEBUG: Searching for Model Path Part='{model_path_part}', Version/Filename='{model_version_tag}'")

    # --- Attempt 1: Library Path ---
    # The version tag IS the filename
    library_path = MANIFEST_BASE_DIR / "registry.ollama.ai" / "library" / model_path_part / model_version_tag
    print(f"    Checking Library path: {library_path}")
    if library_path.is_file():
        print(f"    Found manifest at Library path: {library_path}")
        return library_path

    # --- Attempt 2: HF/Other Path (treat model_path_part as relative root) ---
    # The version tag IS the filename
    other_path = MANIFEST_BASE_DIR / model_path_part / model_version_tag
    print(f"    Checking Other path: {other_path}")
    if other_path.is_file():
        print(f"    Found manifest at Other path: {other_path}")
        return other_path

    # --- Attempt 3: Fallback Search (os.walk) ---
    # Search for a *file* named model_version_tag within a directory matching model_path_part
    print(f"    Constructed paths failed, trying os.walk search...")
    # Construct the expected final parts of the path
    expected_parent_dir_name = Path(model_path_part).name # e.g., 'gemma3' or 'LatexMind-2B...'
    expected_grandparent_dir_name = Path(model_path_part).parent.name # e.g., 'library' or 'mradermacher'

    print(f"    Searching for file '{model_version_tag}' in dir '.../{expected_grandparent_dir_name}/{expected_parent_dir_name}/'")

    found_path = None
    try:
        for root, dirs, files in os.walk(MANIFEST_BASE_DIR):
            current_root = Path(root)
            # Check if the current file list contains our target filename
            if model_version_tag in files:
                # Check if the parent and grandparent directory names match expectations
                if current_root.name == expected_parent_dir_name and current_root.parent.name == expected_grandparent_dir_name:
                    potential_path = current_root / model_version_tag
                    print(f"    Found potential match via walk: {potential_path}")
                    # Verify it's actually a file
                    if potential_path.is_file():
                        found_path = potential_path
                        break # Stop after first match

        if found_path:
             print(f"    Found manifest via os.walk: {found_path}")
             return found_path
        else:
             print(f"    Manifest not found via os.walk.")
             return None
    except Exception as e:
        print(f"    [WARN] Error during os.walk search: {e}")
        return None
# --- END MODIFIED ---


def extract_and_copy_gguf(ollama_tag: str, target_filename: str):
    """Finds manifest, gets largest blob digest, and copies the blob."""
    # ... (Keep the rest of this function the same, it calls the modified find_manifest_file) ...
    print("-" * 40)
    print(f"Processing Ollama Tag: {ollama_tag}")

    target_filepath = TARGET_GGUF_DIR / target_filename
    print(f"  Target llama.cpp path: {target_filepath}")

    if target_filepath.exists():
        print(f"  INFO: Target file '{target_filename}' already exists. Skipping.")
        return True # Indicate skipped

    # Find the manifest file using the revised function
    manifest_path = find_manifest_file(ollama_tag)
    if not manifest_path:
        # Error message printed within find_manifest_file
        print(f"  ERROR: Skipping '{ollama_tag}' due to manifest not found.")
        return False # Indicate failure

    print(f"  Using manifest: {manifest_path}") # Print the path found

    # Read and parse manifest JSON
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest_data = json.load(f)
    except Exception as e:
        print(f"  ERROR: Failed to read or parse JSON from manifest {manifest_path}: {e}")
        return False

    # Find the largest layer digest
    try:
        layers = manifest_data.get('layers', [])
        if not layers:
            print(f"  ERROR: No layers found in manifest {manifest_path}.")
            return False
        layers.sort(key=lambda x: x.get('size', 0), reverse=True)
        gguf_digest = layers[0].get('digest')
        if not gguf_digest:
            print(f"  ERROR: Could not find digest for the largest layer in {manifest_path}.")
            return False
        print(f"  Identified largest layer digest: {gguf_digest}")
    except Exception as e:
        print(f"  ERROR: Failed processing layers in manifest {manifest_path}: {e}")
        return False

    # Construct blob path
    blob_filename = gguf_digest.replace("sha256:", "sha256-")
    source_blob_path = BLOB_DIR / blob_filename
    print(f"  Expected source blob path: {source_blob_path}")

    # Check if source blob exists
    if not source_blob_path.is_file():
        print(f"  ERROR: Source blob file '{blob_filename}' not found in {BLOB_DIR}.")
        print(f"         Manifest ({manifest_path}) points to a non-existent blob?")
        return False

    # Copy the blob
    print(f"  Copying '{source_blob_path}' to '{target_filepath}'...")
    try:
        shutil.copy2(source_blob_path, target_filepath)
        print(f"  SUCCESS: Copied GGUF for tag '{ollama_tag}' to '{target_filename}'.")
        return True
    except Exception as e:
        print(f"  ERROR: Failed to copy blob file for tag '{ollama_tag}': {e}")
        return False


def main():
    """Main function to orchestrate the process."""
    # ... (Keep main function the same) ...
    print("Starting GGUF extraction process using manifest finding (Python)...")
    TARGET_GGUF_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ensuring target directory exists: {TARGET_GGUF_DIR}")
    print(f"Using Ollama models directory: {OLLAMA_MODELS_DIR}")

    if not OLLAMA_MODELS_DIR.is_dir() or not BLOB_DIR.is_dir() or not MANIFEST_BASE_DIR.is_dir():
        print("[ERROR] Critical Ollama directories not found. Exiting.")
        exit(1)

    print("\nFetching installed Ollama models...")
    installed_models_str, err = run_command(['ollama', 'list'])
    installed_model_tags = set()
    if installed_models_str:
        lines = installed_models_str.strip().split('\n')
        if len(lines) > 1:
            for line in lines[1:]:
                 parts = line.split()
                 if parts: installed_model_tags.add(parts[0])
        print(f"Found {len(installed_model_tags)} installed model tags.")
    else: print("[WARN] Could not retrieve installed models list from 'ollama list'.")

    models_processed = 0; models_succeeded = 0; models_skipped = 0; models_failed = 0
    for ollama_tag, target_filename in OLLAMA_TO_LLAMA_CPP_FILENAME.items():
        models_processed += 1
        if ollama_tag not in installed_model_tags and installed_models_str is not None:
             print("-" * 40); print(f"Processing Ollama Tag: {ollama_tag}"); print(f"  WARN: Tag '{ollama_tag}' is mapped but not found in 'ollama list'. Skipping."); models_skipped += 1; continue

        success = extract_and_copy_gguf(ollama_tag, target_filename)
        if success: models_succeeded += 1
        else: models_failed += 1

    print("-" * 40); print("GGUF extraction process finished."); print(f"Summary: Processed={models_processed}, Succeeded/Skipped={models_succeeded}, Failed={models_failed}"); print(f"Please verify the files in: {TARGET_GGUF_DIR}"); print("NOTE: This script relies on finding manifest files in ~/.ollama and assumes the largest layer is the GGUF."); print("      If issues occur, manual inspection or direct download is recommended.")


if __name__ == "__main__":
    main()