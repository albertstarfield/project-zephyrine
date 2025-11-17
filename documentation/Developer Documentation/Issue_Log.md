# Project Zephyrine - Canonical Issue & Requirement Log

This document serves as the master list for all tracked requirements, features, bugs, and test cases for Project Zephyrine. All substantive commits must reference an ID from this log in their footer.

---

## **BUILD**

### Chore

-   **ID:** `BUILD-CHORE-001`
    -   **Title:** Remove conda executable path cache file.
    -   **Status:** `In-Progress`
    -   **Description:** Removed the `.conda_executable_path_cache_main_env.txt` file from the repository. This file is specific to a local user environment and should not be tracked in version control.

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
-   **ID:** `MESH-BUG-005`
    -   **Title:** Fix asset path resolution in ZephyMesh node.
    -   **Status:** `In-Progress`
    -   **Description:** Corrected the path resolution for the `relayDir` and `staticModelPoolPath` in the ZephyMesh node to ensure correct asset discovery.

---

## **DOCUMENTATION (DOC)**

### Requirements

-   **ID:** `DOC-REQ-003`
    -   **Title:** Add academic citations and clarify architectural concepts in README.
    -   **Status:** `In-Progress`
    -   **Description:** The main `README.md` has been updated to include academic citations for the architectural concepts mentioned, such as big.LITTLE and Agentic Context Engineering. The credits and feature descriptions have also been clarified.

-   **ID:** `MESH-DOC-001`
    -   **Title:** Create component architecture and design documentation for ZephyMesh.
    -   **Status:** `Resolved`
    -   **Description:** A formal `README.md` file must be created for the `zephyMeshNetwork` component. This document must outline the core principles, architecture, feature roadmap, and developer notes to serve as the canonical source of truth for the sub-project.

-   **ID:** `DOC-REQ-004`
    -   **Title:** Formalize Issue Log Structure and Content
    -   **Status:** `Resolved`
    -   **Description:** This entry formalizes the structure and content of the `Issue_Log.md` file itself. It ensures that all future entries adhere to the established format of ID, Title, Status, and Description, as defined in the `CONTRIBUTING.md` guidelines. This meta-entry serves as a reference point for the log's own standards.

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

-   **ID:** `CORE-FEAT-012`
    -   **Title:** Add langchain and langchain-text-splitters to dependencies.
    -   **Status:** `In-Progress`
    -   **Description:** Added `langchain` and `langchain-text-splitters` to the project's dependencies to enable new features related to language model chains and text processing.

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

-   **ID:** `CORE-FEAT-008`
    -   **Title:** Unify Mesa GPU Driver Build and Add CPU Fallback for Android.
    -   **Status:** `In-Progress`
    -   **Description:** The Mesa build process for Android containers has been unified to use a single configuration for both Adreno and Mali GPUs. A verification step using `vulkaninfo` has been added to check if the custom driver is loaded correctly. If verification fails, a `.gpu_acceleration_failed` flag is created, forcing the application into a CPU-only fallback mode to ensure functionality on devices where the custom driver fails.

-   **ID:** `CORE-FEAT-009`
    -   **Title:** Implement Casual Mistype Humanizer.
    -   **Status:** `Resolved`
    -   **Description:** Adds a feature to programmatically introduce subtle, human-like errors into the AI's responses to make the persona more believable. This includes chances for lowercase starts, lowercase after periods, capitalization mishaps, and punctuation omissions.

-   **ID:** `CORE-FEAT-010`
    -   **Title:** Pre-buffer 'thinking' monologue on startup.
    -   **Status:** `Resolved`
    -   **Description:** To improve UI responsiveness, the introspective monologue shown when the AI is "thinking" is now generated and cached in the database on startup. This avoids a real-time generation call and the associated delay when the user first interacts with the AI.
-   **ID:** `CORE-FEAT-011`
    -   **Title:** Add ethical watermark for user-invoked image generation.
    -   **Status:** `In-Progress`
    -   **Description:** Implemented an optional, repeating, diagonal, semi-transparent watermark on images generated by user request to discourage misuse and promote ethical AI art generation.

### Defects (Bugs)

-   **ID:** `CORE-BUG-001`
    -   **Title:** Fix unlimited undefined entry of Assistant and User on the database at background_generate result raw ChatML generation.
    -   **Status:** `Resolved`
    -   **Description:** Addressed an issue where `background_generate` was creating an unlimited number of undefined Assistant and User entries in the database due to improper handling of raw ChatML generation results. This fix ensures that only valid and properly attributed entries are stored, preventing database bloat and maintaining data integrity.

-   **ID:** `CORE-BUG-002`
    -   **Title:** Prevent crashes from oversized embedding batches.
    -   **Status:** `Resolved`
    -   **Description:** Added a safeguard in `cortex_backbone_provider.py` to handle cases where a single text item exceeds the maximum token limit for embedding. The code now truncates oversized items before adding them to a batch, preventing the embedding process from crashing and ensuring that large documents can be processed reliably.

-   **ID:** `CORE-BUG-003`
    -   **Title:** GPU Driver Verification Fails on Some Android Devices.
    -   **Status:** `Open`
    -   **Description:** The `vulkaninfo` check during the Mesa build process fails on some Android devices, causing the system to unnecessarily fall back to CPU mode even if the GPU is functional. This appears to be a false negative from the verification step.

-   **ID:** `CORE-BUG-004`
    -   **Title:** Launcher fails to correctly configure environment in proot/Termux.
    -   **Status:** `In-Progress`
    -   **Description:** The launcher script has several issues when running in a proot-ed environment on Termux. It fails to correctly identify the environment, leading to incorrect paths for conda, contaminated miniforge installations, and failure to set up the environment for custom GPU drivers. This commit introduces fixes to make the launcher more robust in these environments.
-   **ID:** `CORE-BUG-005`
    -   **Title:** Fix streaming implementation in LlamaCppChatWrapper.
    -   **Status:** `In-Progress`
    -   **Description:** Removed the `_stream` method from `LlamaCppChatWrapper` as it was incompatible with raw prompt mode and causing errors.

### Refactors

-   **ID:** `CORE-REFACTOR-001`
    -   **Title:** Refactor ToT payload and increase embedding context size.
    -   **Status:** `Resolved`
    -   **Description:** Refactored the Tree of Thought (ToT) payload in `AdelaideAlbertCortex.py` for clarity and simplicity. Increased the default embedding context size in `cortex_backbone_provider.py` to 4096 to improve embedding quality for longer documents. Simplified the frontend input area in `InputArea.jsx` by removing the stop generation button.

-   **ID:** `CORE-REFACTOR-002`
    -   **Title:** Overhaul RAG and background processing for robustness and traceability.
    -   **Status:** `Resolved`
    -   **Description:** Major refactor of the `background_generate` function in `AdelaideAlbertCortex.py`. The RAG pipeline was rebuilt to use a safe, robust helper (`_build_on_the_fly_retriever`) that combines vector and fuzzy search for recent history. The entire background task now logs every intermediate thought, draft, and correction as a distinct, traceable interaction in the database, providing a complete audit trail of the AI's reasoning process. The reflection process was also redesigned to be a "pure" operation that creates new records instead of updating old ones, ensuring data immutability.

-   **ID:** `CORE-REFACTOR-003`
    -   **Title:** Refactor database initialization for speed and robustness.
    -   **Status:** `Resolved`
    -   **Description:** The database initialization process (`init_db`) has been refactored to be significantly faster. It now performs a fast, optimistic initialization and spawns a background thread for slower integrity checks, repairs, and migrations. This allows the application to start much faster. A health check event (`_DB_HEALTH_OK_EVENT`) is used to signal when the database is fully ready.
-   **ID:** `CORE-REFACTOR-004`
    -   **Title:** Refactor Flask route registration from @app.route to @system.route.
    -   **Status:** `In-Progress`
    -   **Description:** Refactored all Flask routes in `AdelaideAlbertCortex.py` to use the `@system.route` decorator instead of `@app.route` for better organization and consistency.

-   **ID:** `CORE-REFACTOR-005`
    -   **Title:** Update dependencies and refactor imports for langchain compatibility.
    -   **Status:** `Resolved`
    -   **Description:** Updated project dependencies and code to align with recent changes in the langchain library. This includes adding `langchain-experimental` to `requirements.txt`, changing the import for `RecursiveCharacterTextSplitter` from `langchain.text_splitter` to `langchain_text_splitters` across multiple files, and updating the `aria2` conda package source in `launcher.py`.

---



## **LAUNCHER**



### Features



-   **ID:** `LAUNCHER-FEAT-001`

    -   **Title:** Implement a fast-path startup sequence ("Death Stranding II Inspired Loading").

    -   **Status:** `Resolved`

    -   **Description:** To significantly reduce application startup time after the initial setup, a "fast path" has been implemented in `launcher.py`. On the first successful run, the launcher now compiles all project Python source code into optimized bytecode (`.pyc` files) and creates a `.setup_complete_v2` flag file. On subsequent launches, if this flag is present, the launcher bypasses all dependency checks and environment verification, launching all services in parallel for a near-instantaneous startup.
-   **ID:** `LAUNCHER-FEAT-002`
    -   **Title:** Prioritize Hypercorn service startup over Watchdog.
    -   **Status:** `In-Progress`
    -   **Description:** Changed the service startup sequence in `launcher.py` to prioritize the Hypercorn server, ensuring the API is available before other services.


### Defects (Bugs)

-   **ID:** `LAUNCHER-BUG-003`
    -   **Title:** Reduce ZephyMesh startup timeout.
    -   **Status:** `In-Progress`
    -   **Description:** The timeout for waiting for the ZephyMesh port file has been reduced from 600 to 60 seconds. The previous timeout was excessively long and could cause the launcher to hang if the mesh node failed to start.

-   **ID:** `LAUNCHER-BUG-001`
    -   **Title:** Fix double launch port occupancy issue.
    -   **Status:** `In-Progress`
    -   **Description:** Fixed an issue where the launcher could attempt to launch a service on a port that was already occupied, causing a crash. The launcher now checks for port availability before launching services.

-   **ID:** `LAUNCHER-BUG-002`
    -   **Title:** Launcher fails to correctly configure environment in proot/Termux and has unstable Python version handling.
    -   **Status:** `In-Progress`
    -   **Description:** The launcher script has several issues. It struggles to correctly identify and configure the environment when running inside a proot-ed Termux environment, leading to incorrect paths and installation failures. The Python version selection logic is too aggressive and not stable, attempting to use the newest versions first instead of prioritizing known-good versions. The script also has hardcoded paths and lacks robustness in handling dependency installation failures. This commit introduces a major overhaul to address these issues by improving environment detection, implementing a "stability-first" Python version selection strategy, and making the dependency installation process more resilient.

---



## **UI ENGINE (UI)**


### Features

-   **ID:** `UI-FEAT-001`
    -   **Title:** Refactor chat submission to synchronous flow and remove typing indicator.
    -   **Status:** `Resolved`
    -   **Description:** Changed the chat message submission from an optimistic UI update to a synchronous flow. The user's message is now displayed only after it has been successfully saved to the database. This provides a clearer guarantee of message delivery. Additionally, the assistant's "typing indicator" dots were removed to simplify the UI.

-   **ID:** `UI-FEAT-002`
    -   **Title:** Display chronological timestamp under chat messages.
    -   **Status:** `Resolved`
    -   **Description:** Implemented the display of the full chronological date, time, and seconds underneath each chat bubble in the chat feed. This provides users with precise timing information for each message.


### Performance



-   **ID:** `UI-PERF-001`

    -   **Title:** Implement virtualized scrolling for chat history to fix performance issues.

    -   **Status:** `Resolved`

    -   **Description:** To improve frontend performance and handle long chat histories efficiently, the chat interface now uses virtualized scrolling (windowing). Instead of rendering all messages at once, it now only renders a small, visible subset. As the user scrolls to the top of the history, older messages are dynamically loaded and rendered on demand. This resolves severe performance degradation and the "bouncy" UI behavior in chats with many messages.



### Defects (Bugs)



-   **ID:** `UI-BUG-001`

    -   **Title:** Page bouncing back on ordinary chat mode.

    -   **Status:** `Resolved`

    -   **Description:** The chat page bounces back to the top on ordinary chat mode, which is a very annoying user experience. This was caused by rendering the entire chat history at once, leading to performance issues.