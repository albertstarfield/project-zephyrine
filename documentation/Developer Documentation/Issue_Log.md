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
    -   **Status:** `Resolved`
    -   **Description:** Implemented functionality to store the full content of indexed documents directly into the database. This enables efficient local document fetching for RAG (Retrieval Augmented Generation) and other content-aware operations, reducing reliance on external file system access during runtime.

-   **ID:** `CORE-FEAT-003`
    -   **Title:** Added MultiLanguage ZH and EN and native lang depends on query summarization for accommodating fallback of fuzzy matching on direct_generate and background_generate connection on the database.
    -   **Status:** `Resolved`
    -   **Description:** Introduced multi-language summarization capabilities, supporting English (EN), Simplified Chinese (ZH), and the original query language. This enhancement improves the robustness of fuzzy matching for `direct_generate` and `background_generate` database connections by providing summarized content in multiple linguistic contexts, facilitating better retrieval and understanding across diverse user inputs.

-   **ID:** `CORE-FEAT-004`
    -   **Title:** Reduce the ELP0 restart chance for each pass.
    -   **Status:** `Resolved`
    -   **Description:** Modified the ELP0 (Elevated Level Privilege 0) interruption retry mechanism to significantly reduce the chance of full system restarts for each pass. This improves system stability and responsiveness by allowing for more graceful handling of transient interruptions during critical operations.

-   **ID:** `CORE-FEAT-005`
    -   **Title:** Enhance search, debugging, and system stability.
    -   **Status:** `Resolved`
    -   **Description:** This update introduces several enhancements across the system. It adds fuzzy and vector search results as an always-on augmented result. More detailed debug logs are inserted into the interaction history, making them accessible to `direct_generate` for learning from mistakes. The `CortexConfiguration` has been adjusted to include a 25% safety context for token counting mismatches, with the context size set to 4096. The file indexer's idle cycle is now 3600 seconds to allow ELP0 and self-reflection to execute. Finally, the interaction indexer now includes non-text data in the augmentation mix.

-   **ID:** `CORE-FEAT-006`
    -   **Title:** Implement Recursive Socratic Inquiry for Self-Correction and Learning.
    -   **Status:** `Resolved`
    -   **Description:** Introduced the A.R.I.S.E. (Adaptive Recursive Inquiry & Synthesis Engine) architecture. This system enables a recursive learning loop where the AI generates Socratic questions based on its own outputs (drafts, summaries, final answers). These questions are saved as new tasks, which the AI then attempts to answer, creating a continuous cycle of self-improvement and knowledge expansion. This includes per-step inquiry generation and a dedicated `socratic_thought` classification.

-   **ID:** `CORE-FEAT-007`
    -   **Title:** Implement Dynamic Agentic Relaxation for Resource Management.
    -   **Status:** `Resolved`
    -   **Description:** The `AgenticRelaxationThread` in `priority_lock.py` now supports a dynamic, resource-aware mode (`reservativesharedresources`). In this mode, the thread monitors system CPU load and ELP1 task contention to aggressively throttle ELP0 background tasks when the system is busy, and allows them to run freely when resources are available. This improves overall system responsiveness.

### Defects (Bugs)

-   **ID:** `CORE-BUG-001`
    -   **Title:** Fix unlimited undefined entry of Assistant and User on the database at background_generate result raw ChatML generation.
    -   **Status:** `Resolved`
    -   **Description:** Addressed an issue where `background_generate` was creating an unlimited number of undefined Assistant and User entries in the database due to improper handling of raw ChatML generation results. This fix ensures that only valid and properly attributed entries are stored, preventing database bloat and maintaining data integrity.

-   **ID:** `CORE-BUG-002`
    -   **Title:** Prevent crashes from oversized embedding batches.
    -   **Status:** `Resolved`
    -   **Description:** Added a safeguard in `cortex_backbone_provider.py` to handle cases where a single text item exceeds the maximum token limit for embedding. The code now truncates oversized items before adding them to a batch, preventing the embedding process from crashing and ensuring that large documents can be processed reliably.

### Refactors

-   **ID:** `CORE-REFACTOR-001`
    -   **Title:** Refactor ToT payload and increase embedding context size.
    -   **Status:** `Resolved`
    -   **Description:** Refactored the Tree of Thought (ToT) payload in `AdelaideAlbertCortex.py` for clarity and simplicity. Increased the default embedding context size in `cortex_backbone_provider.py` to 4096 to improve embedding quality for longer documents. Simplified the frontend input area in `InputArea.jsx` by removing the stop generation button.

-   **ID:** `CORE-REFACTOR-002`
    -   **Title:** Overhaul RAG and background processing for robustness and traceability.
    -   **Status:** `Resolved`
    -   **Description:** Major refactor of the `background_generate` function in `AdelaideAlbertCortex.py`. The RAG pipeline was rebuilt to use a safe, robust helper (`_build_on_the_fly_retriever`) that combines vector and fuzzy search for recent history. The entire background task now logs every intermediate thought, draft, and correction as a distinct, traceable interaction in the database, providing a complete audit trail of the AI's reasoning process. The reflection process was also redesigned to be a "pure" operation that creates new records instead of updating old ones, ensuring data immutability.
