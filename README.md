<h1 align="center">

<sub>
<img src="documentation/ProjectZephy023LogoRenewal.png" height=256>
</sub>
<br>
</h1>

<h5 align="center"> </h5>


<h5 align="center">
<sub align="center">
<img src="documentation/Project%20Zephyrine%20HandDrawnPersonalized%20Logo.png" height=128>

</sub>
</h5>
<p align="center"><i>Hello there! I'm Adelaide Zephyrine Charlotte, Fascinating and a very nice moment to meet you, They usually called me Zephy. Hey are you ready to explore the aether with me?</i></p>

<p align="center"><h5>In Self-learning and Self-improvement We Trust</h5></p>
<hr>

[![Hippocratic License HL3-BDS-BOD-LAW-MEDIA-MIL-SOC-SUP-SV](https://img.shields.io/static/v1?label=Hippocratic%20License&message=HL3-BDS-BOD-LAW-MEDIA-MIL-SOC-SUP-SV&labelColor=5e2751&color=bc8c3d)](https://firstdonoharm.dev/version/3/0/bds-bod-law-media-mil-soc-sup-sv.html)

## Adelaide Lite: The Lightweight version Ada Core of Project Zephyrine

### A Glimpse Into the Aether: Abstract

Adelaide Lite is an highly efficient iteration of Project Zephyrine's core architecture. Written Mostly in **Ada** and built atop the robust `AWS` (Ada Web Server) framework, Adelaide Lite distills the conversational and adaptive nature of Zephyrine into an engine designed for minimal storage footprint, low runtime memory usage, and absolute stability.

This is just another Adaptive Agent wrapper with determenistic response on certain question and query represents the integration of the **Stella-Icarus Deterministic Core** directly into the API routing layer. By utilizing an Elevated Level Privilege (ELP) priority thread scheduler and native OS tasking, we can ensure that deep cognitive reasoning (ELP0) never interrupts high-priority real-time responses or physical telemetry (ELP1/ELP2).

### ⚙️ Core Architecture

Adelaide Lite is specifically tailored for environments with lower storage and constrained runtime memory capacity.

1.  **Ada/SPARK Foundation:** Inspired by the rigorous high-integrity software paradigms (DO-178C, ECSS-E-ST-40C, ECSS-Q-ST-80C), the entire networking, caching, and task-scheduling layer is written in Ada, providing mathematically verifiable safety against buffer overflows, race conditions, and memory leaks.
2.  **WCET Enforcements:** The Worst-Case Execution Time (WCET) manager actively monitors and terminates non-deterministic AI generation if it breaches strict latency budgets, falling back to cached or deterministic responses.
3.  **Hybrid Deno Global Internet Reference Hooks:** Instead of embedding fragile web drivers, Adelaide Lite securely spawns isolated Deno/TypeScript subprocesses (`playwright_scraper.ts`) that utilize stealth-plugin techniques to breach bot-detection challenges and retrieve factual data for the generative core without compromising the stability of the Ada parent process.

### 🎭 API Interfaces

Adelaide Lite maintains compatibility with standard communication dialects to ease integration.

*   **The OpenAI Mask (`/v1/*`):** Full compatibility layer for `/v1/chat/completions` and `/v1/models`. Future implementations will expand this to full stateful Assistant APIs routed directly into the local SQLite memory graph.
*   **The Ollama Mask (`/api/*`):** Direct proxy and multiplexing for local inference endpoints (`/api/chat`, `/api/generate`), allowing Zephy to wrap and manage local models seamlessly.

### 🚀 Prerequisites and Quick Start

Adelaide Lite is designed to be highly portable but requires a specific set of tools to minimize bloat.

#### Requirements
*   **Alire (Ada LIbrary REpository):** Required to resolve Ada dependencies and build the core executable.
*   **Deno:** Required for running the stealth web-scraper sidecars. Deno ensures all JavaScript/TypeScript dependencies are sandboxed and natively cached without requiring a full Node.js installation.
*   **Python 3.10+:** Required as the primary orchestrator for complex embedding mathematical operations and SQLite Knowledge Graph interfacing.
*   **PyMuPDF (`fitz`):** For local document indexing and extraction.

#### Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/albertstarfield/Adelaide_Lite
    cd Adelaide_Lite
    ```

2.  **Run the Initialization Script:**
    The `run.sh` script is designed to handle the Ada compilation, Deno setup, and repository cloning (such as `llama.cpp` and `supertonic`) automatically.
    ```bash
    ./run.sh
    ```

    *The script will automatically fetch Ada Web Server (AWS) packages via Alire, install the Deno Playwright Chromium binaries, and start the local API listener on port `11420`.*

## Warning
> A Warning to Adelaide Lite Users newcomers
> 
> **(Please Read Carefully)**
> 
> This is a highly experimental platform combining efficient software paradigms with generative AI. It is **NOT** a plug-and-play ChatGPT clone. The system expects you to actively monitor CPU bounds, configure ELP priority schedulers, and manage the SQLite knowledge graph.
>
> If you are expecting a system that sacrifices reliability for instant "magic" answers, **Look somewhere else. You have been warned.**

---
<h1 align="center">
<sub>
<img src="documentation/madeFromZephyFoundation.png" height=128>
</sub>
<h5 align="center">
Made with Love, Dreams, and Disciplines.<br>
<br>Snail Works</h5> <br>
Zephyrine Foundation 2023-2026
</h5>
<br>
</h1>
