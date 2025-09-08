# Project Zephyrine - Canonical Issue & Requirement Log

This document serves as the master list for all tracked requirements, features, bugs, and test cases for Project Zephyrine. All substantive commits must reference an ID from this log in their footer.

---

## **GOVERNANCE**

### Requirements

-   **ID:** `GOVERNANCE-REQ-001`
    -   **Title:** Define a formal contribution process and workflow.
    -   **Status:** `Resolved`
    -   **Description:** Establish a clear, documented process for all project contributions. This must include a mandatory standard for Git commit messages to ensure traceability and a rule regarding the project launcher as the source of truth for builds. This is foundational for maintaining long-term project quality and enabling effective collaboration.

-   **ID:** `GOVERNANCE-REQ-002`
    -   **Title:** Enforce architectural principles within the project's code of conduct.
    -   **Status:** `Resolved`
    -   **Description:** Amend the `CODE_OF_CONDUCT.md` to formally include adherence to core architectural principles as an expected standard of behavior. This includes a strict prohibition on committing compiled, architecture-specific binaries to the repository.

---

## **ZEPHYMESH NETWORK (MESH)**


### Requirements

-   **ID:** `MESH-REQ-001.B`
    -   **Title:** Ensure manifest accuracy by resolving asset paths correctly.
    -   **Status:** `Resolved`
    -   **Description:** The manifest generation process must correctly identify and resolve the paths to all local asset directories from the project root to ensure a complete and accurate list of shareable files.

-   **ID:** `MESH-REQ-001.C`
    -   **Title:** Expand the P2P manifest scope to include the Hugging Face cache.
    -   **Status:** `Resolved`
    -   **Description:** The asset manifest generation logic must be updated to scan and include the contents of the `huggingface_cache` directory. This is critical for enabling the peer-to-peer distribution of non-GGUF models, tokenizers, and other dynamically downloaded assets.

### Defects (Bugs)

-   **ID:** `ZM-BUG-004`
    -   **Title:** Incorrect working directory causes asset path resolution failure in the Go node.
    -   **Status:** `Resolved`
    -   **Description:** The ZephyMesh Go node process is being launched with its working directory set to its own subdirectory instead of the project root. This causes all relative path lookups for asset directories (e.g., `staticmodelpool`) to fail, resulting in an empty asset manifest and preventing P2P file sharing. The fix is to set the `cwd` to the project's `ROOT_DIR` in `launcher.py`.

---

## **DOCUMENTATION (DOC)**

### Requirements

-   **ID:** `MESH-DOC-001`
    -   **Title:** Create component architecture and design documentation for ZephyMesh.
    -   **Status:** `Resolved`
    -   **Description:** A formal `README.md` file must be created for the `zephyMeshNetwork` component. This document must outline the core principles, architecture, feature roadmap, and developer notes to serve as the canonical source of truth for the sub-project.

### Test Cases (Verification)

-   **ID:** `DOC-REVIEW-001`
    -   **Title:** Documentation Accuracy & Completeness Review for MESH-DOC-001.
    -   **Status:** `Closed`
    -   **Description:** Verify that the `zephyMeshNetwork/README.md` file is present, well-formatted, and accurately reflects the current design, implementation status, and planned features of the ZephyMesh component.

-   **ID:** `DOC-REVIEW-002`
    -   **Title:** Governance Documentation Review for GOVERNANCE-REQ-001 and GOVERNANCE-REQ-002.
    -   **Status:** `Closed`
    -   **Description:** Verify that the new `CONTRIBUTING.md` and the modified `CODE_OF_CONDUCT.md` clearly and correctly establish the formal contribution process and architectural rules.

---

## **TESTING & VALIDATION (TEST)**

### Test Cases

-   **ID:** `ZM-TEST-007`
    -   **Title:** Verification of Manifest Generation for ZM-BUG-004.
    -   **Status:** `Closed`
    -   **Description:** After applying the fix for `ZM-BUG-004`, relaunch the application and inspect the ZephyMesh node's startup logs. Verify that asset directory scanning is successful, warning messages about missing paths are gone, and the `huggingface_cache` is included in the scan.

---

## **CORE ENGINE (CORE)**

### Features

-   **ID:** `CORE-FEAT-001`
    -   **Title:** Upgrade core embedding model from mixedbread to Qwen3.
    -   **Status:** `Resolved`
    -   **Description:** To improve embedding quality and performance, the core embedding model was upgraded from `mxbai-embed-large-v1` to `Qwen/Qwen3-Embedding-0.6B-GGUF`. This required updating model download links in `launcher.py`, configuration in `CortexConfiguration.py`, and fixing the hardcoded embedding context size in `llama_worker.py` to support the new model's 32k context window. Additional changes were made to `database.py` and `file_indexer.py` to handle the new embedding format, and `requirements.txt` was updated with new dependencies.

-   **ID:** `CORE-FEAT-002`
    -   **Title:** Added content pass into the database for local document fetching.
    -   **Status:** `Open`
    -   **Description:** Implemented functionality to store the full content of indexed documents directly into the database. This enables efficient local document fetching for RAG (Retrieval Augmented Generation) and other content-aware operations, reducing reliance on external file system access during runtime.

-   **ID:** `CORE-FEAT-003`
    -   **Title:** Added MultiLanguage ZH and EN and native lang depends on query summarization for accommodating fallback of fuzzy matching on direct_generate and background_generate connection on the database.
    -   **Status:** `Open`
    -   **Description:** Introduced multi-language summarization capabilities, supporting English (EN), Simplified Chinese (ZH), and the original query language. This enhancement improves the robustness of fuzzy matching for `direct_generate` and `background_generate` database connections by providing summarized content in multiple linguistic contexts, facilitating better retrieval and understanding across diverse user inputs.

-   **ID:** `CORE-FEAT-004`
    -   **Title:** Reduce the ELP0 restart chance for each pass.
    -   **Status:** `Open`
    -   **Description:** Modified the ELP0 (Elevated Level Privilege 0) interruption retry mechanism to significantly reduce the chance of full system restarts for each pass. This improves system stability and responsiveness by allowing for more graceful handling of transient interruptions during critical operations.

### Defects (Bugs)

-   **ID:** `CORE-BUG-001`
    -   **Title:** Fix unlimited undefined entry of Assistant and User on the database at background_generate result raw ChatML generation.
    -   **Status:** `Open`
    -   **Description:** Addressed an issue where `background_generate` was creating an unlimited number of undefined Assistant and User entries in the database due to improper handling of raw ChatML generation results. This fix ensures that only valid and properly attributed entries are stored, preventing database bloat and maintaining data integrity.

### Refactors

-   **ID:** `CORE-REFACTOR-001`
    -   **Title:** Refactor ToT payload and increase embedding context size.
    -   **Status:** `Resolved`
    -   **Description:** Refactored the Tree of Thought (ToT) payload in `AdelaideAlbertCortex.py` for clarity and simplicity. Increased the default embedding context size in `cortex_backbone_provider.py` to 4096 to improve embedding quality for longer documents. Simplified the frontend input area in `InputArea.jsx` by removing the stop generation button.
