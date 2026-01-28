-   **ID:** CORTEX-FEAT-018
-   **Title:** Ground AI in Time with Multi-Calendar Temporal Anchor
-   **Status:** Resolved
-   **Description:** This feature enhances the AI's contextual awareness by injecting a temporal anchor into its prompts. At the start of a task, a string is generated containing the current date and time in multiple formats: Gregorian (Common Era), Holocene (Human Era), Hebrew, and Islamic (Hijri). This provides a rich, multi-faceted sense of the current time, intended to improve the model's ability to reason about and retrieve time-sensitive information. Requires the `convertdate` library.

-   **ID:** UI-REFACTOR-006
-   **Title:** Deprecate Ada Frontface and Integrate Native WebSocket UI
-   **Status:** Resolved
-   **Description:** This major refactor removes the Ada-based `frontfacebackendservice` and replaces it with a direct, high-performance WebSocket connection to the main Python backend (`AdelaideAlbertCortex.py`). A new native ASGI WebSocket handler (`/zepzepadaui`) has been implemented. The frontend has been completely overhauled with a new "liquid glass" design, a floating chat input, improved animations, and a streamlined user experience. A new `ZepZepAdaUI` database table has been introduced to separate UI state from the core interaction log.

-   **ID:** CORTEX-FEAT-006
-   **Title:** Implement Warden Memory Protection for LLM Calls
-   **Status:** In-Progress
-   **Description:** This feature introduces the "Warden" system, a proactive memory management and resource negotiation layer for all LLM calls within the Cortex. The Warden is designed to prevent context overflow and subsequent application crashes by dynamically assessing system memory, probing GGUF model metadata to predict RAM requirements, and selecting an appropriate context size ('bin') for the operation. If the required memory exceeds safe available limits, the Warden will negotiate a smaller context bin. If a prompt is too large for the selected bin, the Warden performs "sandwich truncation," preserving the head and tail of the prompt while flushing the full, original text to the vector database for RAG availability. This ensures both stability under memory pressure and full contextual awareness for the system.

-   **ID:** CORE-REFACTOR-003
-   **Title:** Evolve Snowball architecture to Enaga diff/patch loop and enhance robustness
-   **Status:** Resolved
-   **Description:** This major refactor evolves the 'Snowball' architecture into the 'Snowball-Enaga V10' loop. Instead of simple concatenation, this new architecture uses a code model to intelligently generate and apply diff/patches for each generation step, improving contextual consistency. It also introduces a skeleton repair mechanism using a code model to fix malformed JSON plans. Input handling is now more robust with automatic "sandwich" truncation for very long inputs, flushing the full content to the vector database to ensure complete context is available for RAG. The `llama_worker` embedding parser is now more resilient to varied JSON output from the embedding binary. The launcher now ensures `bash` is installed and has better error handling for `libiconv`.

-   **ID:** STELLA-REFACTOR-002
-   **Title:** Remove CX3 Flight Calculator and Refactor Trickshot Benchmark
-   **Status:** Resolved
-   **Description:** This refactor removes the deprecated CX3 flight calculator (`flight_cx3_core.cpp`, `flight_core_ultimate.cpp`, and their associated compiled objects). The `trickshot_stellaicarus_benchmark_simplearitmatic.py` has been updated to use picoseconds for higher precision, and now dynamically recompiles its C++ core only when the source code changes. The benchmark's output has also been enhanced to include its own source code for better introspection. Jemalloc is now enforced in proot environments for stability, and the miniforge path has been added to the environment.

-   **ID:** CORE-FEAT-016
-   **Title:** Implement Dynamic Agentic Relaxation, Enhanced Ollama Compatibility, and VLM Improvements
-   **Status:** In-Progress
-   **Description:** This feature introduces a comprehensive set of improvements. A new "Dynamic Agentic Relaxation" system has been implemented in the `PriorityQuotaLock` to intelligently throttle background ELP0 tasks based on system state (CPU, power, user idle time), significantly improving responsiveness. Ollama compatibility has been greatly enhanced with new API endpoints (`/api/show`), and major refactoring of the streaming generators (`_ollama_pseudo_stream_sync_generator` and `_stream_openai_chat_response_generator_flask`) to provide live log streaming. The Vision Language Model (VLM) pipeline has been made more robust, with its priority elevated to ELP1 and its context handling improved. The default VLM model has been updated to a higher quality quantization. Additionally, numerous performance and stability improvements have been made to the `llama_worker` and `AdelaideAlbertCortex`.

-   **ID:** LLAMA-FIX-001
-   **Title:** Disable memory mapping in llama_worker for stability
-   **Status:** Resolved
-   **Description:** Disabled memory mapping (`--mmap`) for the llama.cpp subprocess by replacing it with `--no-mmap`. This is to prevent potential memory-related issues and improve stability, especially on systems where mmap might be problematic.
-   **ID:** CORTEX-FEAT-004
-   **Title:** Overhaul RAG by Re-adding Hybrid Search and Advanced Prompt Synthesis
-   **Status:** Resolved
-   **Description:** This feature restores and significantly enhances the Cortex's RAG capabilities by re-introducing fuzzy search, which was previously erroneously removed by an AI agent. The `direct_rag_retriever` now uses a robust hybrid approach, combining high-precision semantic vector search with a fuzzy search fallback to improve context retrieval. A new configuration, `FUZZY_SEARCH_THRESHOLD_CONTEXT`, was introduced for this. Furthermore, the core `PROMPT_DIRECT_GENERATE` was rewritten to give the AI explicit, step-by-step instructions on how to comprehensively synthesize information from retrieved context, leading to more detailed and useful responses. A DeepWiki badge was also added to the `README.md`.
-   **ID:** STELLA-REFACTOR-001
-   **Title:** Refactor Avionics Daemon from Simulator to Deterministic Kernel
-   **Status:** In-Progress
-   **Description:** This refactor transforms the `avionics_daemon` from a random data simulator into a deterministic, physics-based kernel. It introduces a thread-safe state manager and a physics integration loop. Two-way communication with the Python orchestrator is now established via `stdin`/`stdout` pipes, allowing for external control. The Python host's build and discovery logic for Ada daemons has also been significantly improved with better error logging and dynamic executable naming.
-   **ID:** DOCS-UPDATE-001
-   **Title:** Update Hippocratic License badge in README
-   **Status:** Resolved
-   **Description:** Updated the Hippocratic License badge in `README.md` to include the `LAW` and `SOC` clauses, changing it from `HL3-BDS-BOD-MEDIA-MIL-SUP-SV` to `HL3-BDS-BOD-LAW-MEDIA-MIL-SOC-SUP-SV`.
-   **ID:** SYS-PERF-008
-   **Title:** Tune Cortex performance, improve response normalization, and prevent file indexer stalls.
-   **Status:** Resolved
-   **Description:** This change introduces several performance and reliability improvements. In `CortexConfiguration.py`, background task limits, log queue size, and flush intervals were significantly increased to handle larger workloads, while the fuzzy search threshold and reflector idle wait time were lowered to improve responsiveness. New regex-based normalization rules were added to `DIRECT_GENERATE_NORMALIZATION_RULES` to aggressively clean up verbose, conversational filler from AI responses. In `file_indexer.py`, the automatic unloading of VLM models was disabled to prevent a suspected race condition causing the indexer to hang overnight. A blank line was added to `.gitignore` for readability.
-   **ID:** VECTCOMP-PERF-007
-   **Title:** Refactor QRNN simulation to use GPU-accelerated tensor engine
-   **Status:** Resolved
-   **Description:** Majorly refactored `VectorCompute_Provider.py` to replace the memory-intensive, numpy-based dense matrix simulation with a high-performance, memory-optimized tensor contraction engine. The new engine automatically utilizes PyTorch for GPU acceleration (CUDA, ROCm, MPS) if available, falling back to accelerated CPU routines. This change significantly improves simulation speed and reduces the memory footprint by avoiding the construction of large, dense unitary matrices. The implementation now relies on efficient `tensordot` operations and index permutation for entanglement, which is well-suited for modern hardware. The provider now dynamically selects the best backend (PyTorch, Qiskit, etc.) at initialization. Minor logging noise was reduced in `cortex_backbone_provider.py` and `file_indexer.py`.
-   **ID:** CORTEX-REFACTOR-001
-   **Title:** Remove unused datetime import in AdelaideAlbertCortex.py
-   **Status:** Resolved
-   **Description:** Removed an unnecessary `datetime` import from `systemCore/engineMain/AdelaideAlbertCortex.py` to clean up the code.
-   **ID:** CORE-REFACTOR-002
-   **Title:** Streamline DB Init, Improve Scheduler Resilience, and Add Preemption Check
-   **Status:** In-Progress
-   **Description:** This change involves several refactors and a new feature. The database initialization logic was simplified by removing the complex Alembic "stamp and upgrade" process in favor of a more direct `create_all` approach, and the background health check thread was removed. The DCTD scheduler is now more aggressive, attempting to connect to the database immediately upon startup without waiting for a health signal. A new `is_preempted` method was added to the `PriorityQuotaLock` to allow tasks to check if they have been preempted by a higher-priority task. Finally, `datetime` imports were standardized to `import datetime as dt` to prevent import shadowing.
-   **ID:** CORTEX-FEAT-003
-   **Title:** Integrate STEM Compass MoE Models
-   **Status:** Resolved
-   **Description:** This feature integrates a suite of specialized Mixture-of-Experts (MoE) models, collectively named "STEM Compass," into the cortex. The launcher (`launcher.py`) has been updated to download these new GGUF models, and the `CortexConfiguration.py` has been modified to map them for use by the LLM worker. The total model parameter count has been updated to 78.15B to reflect these additions. The `.gitignore` file was also updated to exclude temporary build artifacts.
-   **ID:** SYS-PERF-004
-   **Title:** Tune resource limits, improve logging, and relax dependency pinning
-   **Status:** Resolved
-   **Description:** This change introduces several modifications to improve system performance and stability. It adjusts resource limits in `CortexConfiguration.py` for background tasks, database truncation length, and logging to reduce memory footprint and increase responsiveness. Logging is now more frequent with smaller batch sizes. Dependency versions in `requirements.txt` have been relaxed from `==` to `>=` to allow for more flexible package updates. Error handling in the `file_indexer.py` has been improved with a try-except block for image processing. `ENABLE_DB_DELETE_DEFECTIVE_ENTRY` has been enabled to allow for automated cleanup of defective database entries.
-   **ID:** UI-REFACTOR-005
-   **Title:** Remove unused streaming timer logic in ChatPage
-   **Status:** In-Progress
-   **Description:** Refactored the `ChatPage.jsx` component to remove the `streamingStartTimeRef` and the related tokens-per-second calculation. This logic was unused and added unnecessary complexity to the component's state management during message streaming.
-   **ID:** VECTCOMP-REFACTOR-006
-   **Title:** Isolate QRNN Numpy Simulation in Subprocess for Memory Safety
-   **Status:** Resolved
-   **Description:** Refactored the `VectorCompute_Provider.py` to execute the memory-intensive QRNN numpy simulation in an isolated subprocess. This prevents the main application from crashing due to excessive memory allocation during large matrix operations (`2^16 x 2^16` matrices). The core logic is now in a standalone `_execute_numpy_qrnn_isolated` function, and the provider orchestrates the subprocess lifecycle, passing data via temporary files. This guarantees that memory used by the simulation is released back to the OS immediately upon completion.
-   **ID:** CORE-FEAT-015
-   **Title:** Refactor Generation Logic to Peer Review Architecture and Improve Priority Lock
-   **Status:** In-Progress
-   **Description:**
    *   Refactors the primary `chat_direct_generate` function to a "Peer Review Everphase Context" (V9) architecture. This involves a loop of generation, fact-checking against RAG, and routing to specialist models for complex or incomplete responses.
    *   Implements a "politeness" policy in the `PriorityQuotaLock` to prevent ELP0 tasks from starving waiting ELP1 tasks, improving high-priority task responsiveness.
    *   Dele tes the legacy `trickshot_simple_flight_computer.py`.
    *   Increases RAG context limits (`RAG_URL_COUNT`, fuzzy search interaction fetch).
    *   Adjusts logging and disables the "mistype" feature.
-   **ID:** CORTEX-FEAT-005
-   **Title:** Integrate native Llama.cpp multimodal VLM for image processing
-   **Status:** Resolved
-   **Description:** This feature integrates the new `llama-mtmd-cli` (renamed to `LMMultiModal`) from `llama.cpp` for native, high-performance vision language model (VLM) execution. This replaces the previous, less efficient VLM handling. A new `LlamaCppVisionWrapper` and a dedicated `vision` task type in the `llama_worker` have been implemented. The `launcher` and `CortexConfiguration` have been updated with the new Unsloth Qwen VLM models and their required `mmproj` files.
-   **ID:** CORE-REFACTOR-001
-   **Title:** Refactor VLM image description pipeline
-   **Status:** Resolved
-   **Description:** Centralized the Vision Language Model (VLM) image description logic into the `llama_worker.py` script. This removes the `get_vlm_description` method from `cortex_backbone_provider.py` and introduces a dedicated "vision" task type in the worker. The changes also include improved error handling, input processing, and a longer timeout for Ollama streaming to enhance stability with slower models.
-   **ID:** UI-FEAT-005
    -   **Title:** Add initial GTK C application for native UI experiments.
    -   **Status:** In-Progress
    -   **Description:** This adds the initial directory structure and boilerplate for a GTK-based desktop application written in C. This is an experimental feature to explore native UI options for the UIEngine.
-   **ID:** `SYS-REFACTOR-011`
-   **Title:** Overhaul UI architecture, service naming, and persona.
-   **Status:** `Resolved`
-   **Description:** This commit introduces a major refactoring across the system. The frontend UI has been significantly reworked, replacing virtualized scrolling with a more direct rendering approach and implementing a more robust "fire-and-forget" WebSocket messaging architecture. The splash screen sequence was redesigned to include a warning. Service names within the launcher (`launcher.py`) have been standardized (e.g., `ZEPHYMESH-NODE` to `zephymeshHand`). The core assistant persona and system prompts in `CortexConfiguration.py` have been updated to be less confident and more cautious.
