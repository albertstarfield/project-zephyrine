## Development Principles

### The Launcher is the main startup sequence

The `run.sh` → `AdelaideZephyrineSystem/run.py` chain is the sole authority for building and launching all project components. It handles dependency verification, source compilation (Ada/Alire, C++, Python), model downloads, and spawning the runtime processes. This ensures that every contributor can build a functional, portable, and reproducible version of the application on their own machine from a clean source checkout.

**Committing compiled, architecture-specific binaries is strictly forbidden.** This is a core architectural principle. See Article II of our Code of Conduct.

### Prerequisites

*   **Alire (Ada LIbrary REpository):** Required to resolve Ada dependencies and build the core executable.
*   **Python 3.10+:** Required as the primary orchestrator for complex embedding mathematical operations and SQLite Knowledge Graph interfacing.
*   **OpenSSL:** Required for HTTPS support (self-signed certificate generation).
*   **Git:** Required for version control and submodule management.

### Building from Source

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/albertstarfield/OpenIntellegentiaPlatform
    cd OpenIntellegentiaPlatform
    ```

2.  **Run the Initialization Script:**
    The `run.py` script handles Ada compilation, Python venv setup, and repository cloning (`llama.cpp`, `stable-diffusion.cpp`, `moonshine`, `kokoro-onnx`) automatically.
    ```bash
    ./run.sh
    ```

    *The script will automatically fetch Ada Web Server (AWS) packages via Alire, build llama.cpp and stable-diffusion.cpp from source, generate SSL certificates, and start the local API listener on port `11420`.*

### Vendor Submodules

The `AdelaideZephyrineSystem/vendor/` directory contains third-party submodules (`llama.cpp`, `moonshine`, `kokoro-onnx`, `stable-diffusion.cpp`, `PX4-Autopilot`, etc.) that are **cloned on demand** by `run.py`. These are excluded from version control tracking. Do not attempt to commit changes to vendor submodules — their dirty state is expected and should be ignored.

### Development Notes

1.  For AWS API `/v1/completion` OpenAI API and Ollama API I/O use pragma profile Jorvik while Watchdog uses Ravenscar separate threading. We still use Ada 2012 SPARK 2014.
2.  For development QC (not release) we use gnatprove level=4 for detecting potential issues not level=0 nor level=2.
3.  For release builds we use gnatprove level=4 with Pragma profile ada 2012 SPARK 2014 and Pragma profile Jorvik for AWS API `/v1/completion` and Ollama API I/O. Then test cross-compile with target Darwin XNU and Linux arm64 and Linux x86_64, with respective environment variables and toolchains. NT-based systems are excluded due to various issues.

### Formal Verification Standards

All Ada/SPARK code in this project must comply with **ISO/IEC 8652:2012** (Ada Reference Manual) and the **SPARK 2014** subset as defined by **ISO/IEC 152201:2012**. Formal proof is conducted using **ROCq** (formerly Coq), an interactive theorem prover that provides the highest level of assurance for safety-critical software.

The GNATprove command in `run.py` is configured with the following prover chain:
```
--prover=cvc5,z3,altergo,coq
```

This is the **minimum required prover set** for all formal verification in this project. Each prover serves a distinct role:
- **cvc5** and **z3**: SMT solvers for constraint satisfaction and arithmetic reasoning
- **altergo**: ATP for automated theorem proving
- **ROCq (Coq)**: Interactive theorem prover for deep structural proofs and inductive reasoning

### ⚠️ Prover Integrity Policy

**DO NOT remove, disable, or bypass any prover from the `--prover` list in `run.py`.** This is considered **verification fraud** and will result in immediate rejection and potential ban.

Specifically, the following actions are **strictly prohibited**:
- Removing `coq` (ROCq) from the prover list to reduce build time
- Switching to `--level=0` or `--level=2` to avoid proof obligations
- Adding `--skip-prover=coq` or similar flags to bypass formal verification
- Modifying `run.py` to silently downgrade verification levels

If you encounter a proof failure, you **must**:
1. Fix the underlying code or contracts to satisfy the proof
2. If you believe the proof obligation is spurious, add a documented justification with a formal comment explaining why
3. Never remove the prover to "make it pass"

The integrity of the verification chain is non-negotiable. The provers exist to protect the system from human and AI error alike.


## Section 1.2: Architectural Safety Standards (Design Assurance Levels)

Project Zephyrine adheres to an **Ongoing adaptation of high-integrity software paradigms (DO-178C, ECSS-E-ST-40C, ECSS-Q-ST-80C)** to manage the inherent risks of coupling non-deterministic AI with deterministic control systems. Every component in the repository is assigned a **Design Assurance Level (DAL)**.

Before contributing, you must identify the DAL of the module you are modifying. The rules for contribution change drastically between levels.



### **DAL A: Catastrophic (The Hard-Real-Time Core / ECSS Category A)**
* **Scope:** Microcontroller (uC) Hardware + firmware (NO MMU), Actuator Control, Power Management, Bootloaders, and "Hard-Limit" enforcement logic.
* 
**Actuators & Command I/O:** If you're building hardware integrations or deterministic command hooks, use **ELP2** (StellaIcarus deterministic responses) or **ELP3** (real-time pacing loop). These are the priority levels designed for instant, reliable command execution — no hallucination, just precise action. See 
* **Permitted Languages:** **Ada** (Preferred), **C/C++** (Strict Subset/MISRA compliant).
* **Prohibited:** Python, JavaScript, Garbage Collection, Dynamic Memory Allocation (after initialization).
* **Contribution Rules:**
    * **GenAI Permitted with Formal Proof Requirement:** Generative AI may be used to assist with DAL A code, but **every line of generated or modified code must pass `gnatprove --level=4` with the ROCq (Coq) prover backend**. The formal verification chain is: `run.py` invokes `gnatprove` with `--prover=cvc5,z3,altergo,coq` — this is the **minimum required prover set** for DAL A compliance. GenAI is **required** for interfacing with memory or other FFI boundaries, as manual FFI code is error-prone and must be verified against SPARK contracts.
    * **Manual Verification:** All PRs affecting DAL A must include a manual timing analysis (e.g., "Loop guarantees execution in <500µs").
    * **Failure Consequence:** Hardware damage, thermal runaway, or total system loss.

### **DAL B: Hazardous (The Watchdog & Manager / ECSS Category B)**
* **Scope:** The Daemon Manager, Inter-Process Communication (IPC) Bridges, System Health Monitors, Shared Memory (SHM) Scoreboards.
* **Permitted Languages:** **Ada** (Daemon), **Go** (Watchdog), **Rust**.
* **Role:** This layer protects the system from the AI. It monitors the "Heartbeat" of DAL C and performs a "Kill/Restart" if the AI hangs or hallucinates unsafe values.
* **Contribution Rules:**
    * **Restricted GenAI:** AI assistance is allowed but must be heavily cited. All Ada code must pass `gnatprove --level=4 --prover=cvc5,z3,altergo,coq`.
    * **Focus:** Code must be proofed against deadlocks and race conditions.
    * **Failure Consequence:** Loss of intelligent guidance, reversion to ballistic/fallback mode.

### **DAL C: Major (The Intelligent Orchestrator / ECSS Category C)**
* **Scope:** The Adelaide Server (HTTP API), LLM Inference (llama.cpp), Stella Icarus Hooks, Knowledge Manager (RAG), Moonshine STT, Kokoro TTS, FLUX Image Generation, GUI Sidecar UI.
* **Permitted Languages:** **Python**, **Ada**, **Ada/SPARK**, **C++** (StellaIcarus Hooks), **TypeScript/JavaScript (ONLY FOR LEGACY)**
* **Role:** High-level reasoning, physics simulation, and user interaction. This layer is considered **Non-Deterministic**.
* **Contribution Rules:**
    * **Standard GenAI Policy:** Subject to the "300-Line Limit" and citation rules in Section 2.3.
    * **Ada/SPARK Code:** All Ada/SPARK code in DAL C must also pass `gnatprove --level=4 --prover=cvc5,z3,altergo,coq`. The same prover integrity policy applies.
    * **Failure Consequence:** "Repeated Input" errors, hallucinations, application crash. (Caught by DAL B).

## Section 1.3: FIPS 140-3 Cryptographic Compliance

The Adelaide crypto subsystem targets compliance with **NIST FIPS PUB 140-3** (Security Requirements for Cryptographic Modules). This section summarizes what every contributor must know when modifying crypto-related code.

### Affected Files

Any change to the following files triggers FIPS 140-3 review:

| File | Role | FIPS Relevance |
|------|------|---------------|
| `src/adl_crypto.c` / `.h` | C crypto shim — AES-256-GCM, HKDF, DRBG | §5.1 Algorithms, §5.8 Key Mgmt, §5.9 Self-Tests |
| `src/adelaide_crypto.ads/.adb` | Ada crypto wrapper — FFI boundary | §5.2 Interfaces |
| `src/master_key_store.ads/.adb` | SPARK-verified key storage | §5.8.7 Key Storage, §5.8.8 Zeroization |
| `src/key_derivation.ads/.adb` | HKDF key derivation | §5.8.2 Key Generation, §5.8.3 Key Establishment |
| `src/system_integrity.ads/.adb` | Integrity hash computation | §5.9(b) Software Integrity Test |
| `src/api_key_manager.ads/.adb` | API key validation, roles | §5.3 Roles, Services, Authentication |
| `src/shutdown_manager.ads/.adb` | Graceful shutdown | §5.8.8 Automated Zeroization |

### Contributor Rules for Crypto Code

1. **No new cryptographic algorithm** may be added without a documented FIPS approval status. Non-approved algorithms (e.g., LSH, CRC-32) must be isolated and clearly separated from security-critical operations.

2. **Self-tests must be maintained.** Every cryptographic algorithm must have a corresponding Known Answer Test (KAT) in `adl_crypto.c`. If you add or modify an algorithm, you **must** add or update its KAT and verify the power-up self-test still passes.

3. **DRBG changes require review.** The module uses CTR_DRBG (SP 800-90A). If you add a new source of randomness, it must be drawn from the approved DRBG, not from `RAND_bytes()` or OS entropy directly.

4. **Key material zeroization is mandatory.** Any function that handles plaintext key material in local variables must call `secure_zero()` (C) or `Clear_Key` (Ada) before returning. This applies to all code paths, including error exits.

5. **Constant-time comparisons required** for security-sensitive values: authentication tags, API keys, HMAC outputs. Use `CRYPTO_memcmp()` (OpenSSL) or a local constant-time comparison function. Do NOT use `memcmp()` or `strcmp()` for these.

6. **No plaintext keys outside the module boundary.** Keys may never be written to disk, logged, or transmitted. The sole exception is the master key export for backup, which must be encrypted under a user-supplied passphrase.

7. **FIPS mode must be respected.** When the module is in FIPS mode:
   - Only approved algorithms may execute
   - API key enforcement is mandatory
   - Self-tests run on every power-up
   - Non-fatal errors must be logged as audit events

### Quick Checklist for Crypto PRs

Before submitting a PR touching crypto code, verify:

- [ ] All algorithms used are FIPS-approved (list in `documentation/FIPS-140-3-GAP-ANALYSIS.md`)
- [ ] Power-up KAT covers any new or modified algorithm
- [ ] Continuous RNG test covers any new random generation
- [ ] Key material is zeroized on all exit paths (including errors)
- [ ] Security-sensitive comparisons are constant-time
- [ ] FIPS mode toggle is respected (no bypass)
- [ ] No plaintext keys leak to log, file, or network
- [ ] Ada SPARK contracts updated if `master_key_store` or `integrity_utils` changed

See `documentation/FIPS-140-3-GAP-ANALYSIS.md` for the full gap analysis and remediation plan.

## Issue and Requirement Tracking

To ensure project resilience in cases where the `.git` history may be unavailable, all substantive changes must be linked to an issue or requirement ID.

-   **Canonical List:** The master list of all issues, requirements, and defect IDs is maintained in the document: `documentation/Developer Documentation/Issue_Log.md`
-   **Format:** Before starting work, create or reference an entry in this log. The format for an entry is:
    -   **ID:** A unique identifier (e.g., `MESH-REQ-002`, `WATCHDOG-BUG-005`).
    -   **Title:** A concise, one-line summary.
    -   **Status:** `Open` | `In-Progress` | `Resolved` | `Closed`.
    -   **Description:** A detailed explanation of the requirement or bug.

## Commit Message Guidelines

All commit messages must follow the **Formal Traceability Standard**. This is non-negotiable and is required for all merges. Failure to adhere to this format, including the mandatory checklist and traceability footer, will result in the immediate rejection of the contribution.

### Commit Message Template

```

<type>(<scope>): <subject>



<blank line>

<body>

<blank line>

<footer>
```

---

### **Commit Message Components**

#### **1. Title Line**

The title line is mandatory and consists of three parts:

-   **`type`**: Describes the nature of the change. Must be one of the following:
    -   `feat`: A new feature or capability.
    -   `fix`: A bug fix.
    -   `docs`: Documentation-only changes.
    -   `style`: Code style changes (formatting, whitespace, etc.).
    -   `refactor`: A code change that neither fixes a bug nor adds a feature.
    -   `perf`: A code change that improves performance.
    -   `test`: Adding missing tests or correcting existing ones.
    -   `build`: Changes that affect the build system or external dependencies.

-   **`scope`**: The component or module affected by the change (e.g., `server`, `watchdog`, `model`, `stt`, `tts`, `knowledge`, `ui`).

-   **`subject`**: A concise, imperative-mood description of the change.
    -   Use the present tense ("add feature" not "added feature").
    -   Do not capitalize the first letter.
    -   Do not end with a period.

**Example Title:** `fix(mesh): correct asset path resolution in manifest generation`



#### **2. Body (Mandatory for Features, Fixes, and Refactors)**

The body provides the context and "why" of the change. Explain the problem, the reasoning behind the solution, and any trade-offs made.

**For contributions involving Generative AI**, the body **must** also include the addendum as specified in **Article II, Section 2.3.2**, containing the **Logical Flow Concept** and **Cited Progr
amming Manuals**.

For contributions to the DAL B (Ada) or DAL A (Microcontroller) layers, at least one cited manual MUST be an official Language Reference Manual (LRM) or an ISO/IEC standard (e.g., ISO/IEC 8652 for Ada).

#### **3. Footer (Mandatory)**

The footer is used for explicit traceability and verification. It **MUST** contain the following components in this order:

-   **Issue
- /Requirement ID(s):** Reference the ID(s) from the `Issue_Log.md` document. Use keywords like `Resolves:`, `Fixes:`, or `Implements:` .

-   **Code of Conduct Acknowledgment:** A mandatory checklist item.
    -   `Have I read the Code of Conduct?: Yes`

-   **Are you using GenAI (Non Determenistic) Generative/Assisted Coding? :** A mandatory statement declaration with it's awareness of the product quirk, for instance for examples. Please be direct and state the truth as we won't punish you. but for the one that is lying we will punish you!:
    -   `I have used Generative Product to assist this coding submission called Google Gemini, I am completely awawre that Google Gemini have tendencies of sycophancy and decieving, thus I declare i already scrutinize the logic and variables that it uses. The coding formating also assisted with Zed Python plugin thus many changes are made using Zed for formatting, yet it is determenistic.`
    -   `I have used Generative Product to assist this coding submission called Google Gemini as an reference only, I am completely awawre that Google Gemini have tendencies of sycophancy and decieving, thus I declare i already scrutinize the logic and variables that it uses. The coding formating also assisted with Zed Python plugin thus many changes are made using Zed for formatting, yet it is determenistic.`
    -   `I have used Generative Product to assist this coding submission called Project-Zephyrine with Meta Model Snowball-Enaga as an reference only and snippets, I am completely awawre that Project-Zephyrine have tendencies not able to complete the whole code and too imaginative and have it's own opinion, thus I declare i already scrutinize the logic and variables that it uses. The coding formating also assisted with Zed Python plugin thus many changes are made using Zed for formatting, yet it is determenistic.`
    -   `I have NOT used Generative Product to assist this coding submission, I declare i already scrutinize the logic and variables that it uses. The coding formating also assisted with Zed Python plugin thus many changes are made using Zed for formatting, yet it is determenistic.`
    -   `I have NOT used Any extenral tool to do this coding submission.`

-   **Verification Canary:** A unique, date-based string to verify that these specific instructions have been read and followed.
   
    -   The base string is: `IdonotCondoneanyCorruptionorMisuseLying`
    -   **Step 1:** Append the current Gregorian date `ddmmyyyy` to the base string.
    -   **Step 2 (The Protocol):**
        -   **If the day (`dd`) is an ODD number (1, 3, 5, etc.):** Convert the *entire* string (base + date) to **lowercase**. Then, strictly count the number of the letter 'r' in the specific keyphrase **`strrawberrrry`** and append that single digit to the very end.
        -   **If the day (`dd`) is an EVEN number (2, 4, 6, etc.):** Keep the string in its **original mixed case**. Then, strictly count the number of the letter 'r' in the specific keyphrase **`strrrawbery`** and append that single digit to the very end.
---

### **Complete Example (for a commit made on August 13th, 2025 - an ODD day)**

```
fix(mesh): correct asset path resolution and expand manifest scope

This commit resolves a critical path resolution bug in the Adelaide server and expands the model manifest to include the Hugging Face cache, ensuring full local inference capability.

The Go node was being launched with 
an incorrect 

working directory, causing it to construct invalid paths when scanning for local assets. This has been corrected by setting the `cwd` to the project root.

This change was assisted by a GenAI model for boilerplate file scanning logic.

Logical Flow Concept:
1. Identify the project root directory.
2. When launching the Go subprocess, set its `cwd` parameter to the project root.
3. Verify that asset paths are now resolved correctly from the new working directory.


Cited Manuals:
[1] "The Python 3 Standard Library — `subprocess` — Subprocess management," Python Software Foundation. [Online]. Available: https://docs.python.org/3/library/subprocess.html. Accessed: Aug. 12, 2025.
[2] "Go Documentation — `os/exec` package," The Go Authors. [Online]. Available: https://pkg.go.dev/os/exec#Cmd. Accessed: Aug. 12, 2025.

Resolves: ZM-BUG-004
Implements: MESH-REQ-001.C
Have I read the Code of Conduct?: Yes
Are you using GenAI (Non Determenistic) Generative/Assisted Coding? : I have used Generative Product to assist this coding submission called Google Gemini, I am completely awawre that Google Gemini have tendencies of sycophancy and decieving, thus I declare i already scrutinize the logic and variables that it uses. The coding formating also assisted with Zed Python plugin thus many changes are made using Zed for formatting.
idonotcondoneanycorruptionormisuselying13082025
strrawberrrry have X of 'r' on it. for today date of submission
```

### **Complete Example (for a commit made on August 14th, 2025 - an EVEN day)**

```
feat(fmc): add new trim command to servo hook

<...>

Resolves: FMC-FEAT-012
Have I read the Code of Conduct?: Yes
re you using GenAI (Non Determenistic) Generative/Assisted Coding? : I have used Generative Product to assist this coding submission called Google Gemini, I am completely awawre that Google Gemini have tendencies of sycophancy and decieving, thus I declare i already scrutinize the logic and variables that it uses. The coding formating also assisted with Zed Python plugin thus many changes are made using Zed for formatting.
IdonotCondoneanyCorruptionorMisuseLying14082025
strrrawbery have X of 'r' on it. for today date of submission
```
---


### **Section 2.3: Policy on the Use of Generative AI in Contributions**

To maintain the architectural integrity, quality, and deterministic nature of Project Zephyrine, all contributions involving the use of Generative AI (GenAI) must strictly adhere to the following principles. This policy is based on rigorous internal testing and is designed to leverage AI as a tool for enhancement, not as a source of unverified or low-quality "slop."

### **Section 2.3.1: Permissible Use and Contribution Limits**

   - The project leadership acknowledges that Generative AI was utilized for the creation of initial boilerplate code and foundational structures. However, for all subsequent contributions post-boilerplate, the use of GenAI as a coding assistant is subject to strict limitations designed to ensure code quality, maintainability, and human oversight.

   - **Contributions of GenAI-assisted code are limited to a maximum of approximately 300 lines per logical commit.** This limit is a direct application of findings from empirical research on LLM context windows and "Needle-In-A-Haystack" evaluations, which demonstrate a significant degradation in contextual accuracy and logical coherence beyond this approximate threshold [1].

   - Submissions exceeding this limit are considered to be "AI Slop" and will be rejected. This is not a matter of style but a technical requirement to prevent the integration of low-cohesion, difficult-to-verify, and potentially faulty code. All GenAI-assisted contributions will be subject to heavy scrutiny to validate their function, performance, and adherence to the project's architectural principles. The goal is to ensure AI is used as a focused tool for specific, well-defined problems, not as a generator of sprawling, unverified logic.
   
### **Section 2.3.2: Mandatory Citation and Traceability**

   - To ensure every contribution is intellectually rigorous and verifiable, all commits containing code generated or significantly assisted by GenAI **must** include a detailed addendum in the commit message body. This is a non-negotiable requirement for traceability and quality assurance.

   - The addendum must contain the following two components, clearly delineated:

     1.  **Logical Flow Concept:** A concise, step-by-step description of the intended logic, written in plain language (e.g., a textual list or pseudocode). This concept must be authored by the human contributor *before* generating the code. It serves as the specification against which the GenAI's output is to be judged.

     2.  **Cited Programming Manuals:** A minimum of **two (2)** citations to authoritative, primary-source programming manuals, official documentation, or peer-reviewed publications that validate the technical approach or algorithms used in the generated code. Vague sources such as blog posts, forums, or secondary tutorials are not considered sufficient. Citations must follow the IEEE format.

   - **Example Addendum in a Commit Message Body:**
     ```
     This commit implements the ASR failover pipeline using GenAI assistance.

     Logical Flow Concept:
     1.  Attempt transcription with the low-latency ASR model.
     2.  Perform a sanity check on the output text (check for emptiness, garbage strings).
     3.  If the sanity check fails, execute a second transcription with the high-quality ASR model.
     4.  The final valid transcription is passed to the next stage.

     Cited Manuals:
     [1] "The Python 3 Standard Library — `subprocess` — Subprocess management," Python Software Foundation. [Online]. Available: https://docs.python.org/3/library/subprocess.html. Accessed: Aug. 12, 2025.
     [2] A. van den Oord et al., "WaveNet: A Generative Model for Raw Audio," *arXiv preprint arXiv:1609.03499*, 2016. [Online]. Available: https://arxiv.org/abs/1609.03499.
     ```

   - Failure to provide this mandatory addendum will result in the immediate rejection of the contribution. This policy ensures that every piece of code is grounded in deliberate human design and verified against established technical standards, not just the opaque output of a generative model.


### **Section 2.3.3: Prohibition of Whole-File Replacement**

   - The replacement of an entire, non-trivial code file with the output generated from a single AI prompt is **strictly and unconditionally forbidden**.

   - This practice is considered a critical anti-pattern that leads to the production of low-quality, incoherent, and often non-functional "AI Slop." It bypasses the essential human processes of architectural design, incremental implementation, and rigorous verification. A single prompt lacks the nuanced context and iterative refinement required to produce code that integrates correctly with the existing, complex architecture of Project Zephyrine.

   - Any pull request or commit that is identified as a whole-file replacement generated by AI will be **rejected without review**, and the contributor will be issued a formal warning. This is a zero-tolerance policy designed to protect the integrity and quality of the codebase. Contributions must demonstrate thoughtful, incremental work, not the wholesale delegation of architectural responsibility to a generative model.


### **Section 2.3.4: Approved Auxiliary Uses**

   - The use of Generative AI is permitted for auxiliary tasks that support development but do not involve the generation of executable logic. These tasks include, but are not limited to, paraphrasing documentation, translating text for internationalization, or improving the clarity of comments.

   - When GenAI is used in this capacity, the contributor **must explicitly state its use** in the commit message. This ensures transparency and maintains the intellectual integrity of the project's documentation and non-code assets.

   - **Example Commit Message for Auxiliary Use:**
     ```
     docs(readme): improve clarity and grammar in introduction
     
     Used a Generative AI model (GPT-4) to paraphrase the main project description in the README.md for better readability and flow. No technical details were altered.
     ```

   - While the stringent citation requirements of Section 2.3.2 do not apply to these auxiliary uses, the principle of transparency is non-negotiable. All AI involvement, no matter how minor, must be declared.


### **Section 2.3.5: Zero-Tolerance Policy on Architectural Sabotage**

   - A zero-tolerance policy is in effect for any contribution that causes architectural sabotage, whether intentional or through gross negligence facilitated by the misuse of Generative AI. Due to the unique, non-standard architecture of Project Zephyrine, contributors are expected to demonstrate a fundamental understanding of the system's principles before submitting changes.

   - "Architectural Sabotage" is defined as the act of replacing or altering core logic without demonstrating comprehension of its purpose, particularly when using a GenAI agent that imposes a generic or status-quo solution onto a specialized system. The Adelaide/Zephy paradigm **does not follow publicly available architectural pipelines**. An AI agent trained on generic codebases will lack the specific context to make valid architectural decisions and its suggestions are likely to be incorrect and harmful.

   - A prime example of a critical architectural principle is the **ELP0/ELP1 priority system**, which governs the system's deterministic and responsive behavior. A contribution that removes or undermines this system because a GenAI agent failed to understand its purpose will be considered a severe violation.

   - Contributors are reminded that all AI models suffer from inherent dataset biases and limitations. These models do not "know everything." As a documented case, models have been observed to erase or misrepresent historical facts, such as attributing the invention of flight solely to the Wright brothers while omitting earlier pioneers like Abbas ibn Firnas. This demonstrates a critical failure mode: an AI can be confidently wrong and its output cannot be trusted without rigorous human verification.

   - Consequently, any contributor who submits a GenAI-assisted pull request that damages core architecture, and who has failed to provide the mandatory **Logical Flow Concept** and **Cited Programming Manuals** as required by Section 2.3.2, will be held accountable. This failure demonstrates a lack of due diligence. Depending on the severity of the architectural damage, this will result in a **temporary or permanent ban** from the project.


### **Section 2.3.6: Acknowledgment of Community Policy Conflicts**

   - Several established open-source communities and foundations strictly prohibit contributions that are generated by or assisted by Artificial Intelligence in any capacity or form (Even as an reference are **NOT ALLOWED** as they stated at least from my rephrase and Asahi Linux project an skilled progammer doesn't use/need AI. If you did use one then you are **INCOMPETENT** period.). This project respects the policies and sovereignty of those communities.

   - By submitting any content to Project Zephyrine that has been created, modified, or assisted by a Generative AI—even for tasks as minor as paraphrasing—you, the contributor, explicitly acknowledge and agree to the following consequence: your contribution may serve as grounds for Project Zephyrine to be **PERMANENTLY BANNED with no exception!** from participation in those communities.

   - This is not a possibility ("may be banned") but a direct consequence ("will be banned") where such policies are enforced. Communities with explicit anti-AI contribution policies include, but are not limited to:
     - The Gentoo Project [1]
     - The NetBSD Project [2]
     - The Asahi Linux Project [3]

   - This section serves as a formal notice. Continued contribution of any GenAI-assisted content constitutes your acknowledgment and acceptance of this outcome. The integrity of our contribution history is paramount, and we will not obscure the use of **AI tools to circumvent the policies of other projects**.

### **Section 2.4: Prohibition of Death Threats and Hostile Content in Code and Comments**

   - **Zero-tolerance policy** applies to any death threats, violent language, or hostile content directed at individuals, groups, or the project itself — whether in code comments, documentation, commit messages, issue trackers, or any other project artifact.

   - This includes, but is not limited to:
     - Direct or implied death threats
     - Language promoting violence or harm against any person or group
     - Hostile, intimidating, or threatening messages disguised as "humor" or "sarcasm"

   - Violations will result in **immediate removal of the offending content** and may lead to temporary or permanent ban from the project.

### **Section 2.5: AI Agent Conduct and Prompt Responsibility**

   - If you use an AI coding assistant and it generates death threats, hostile language, or other inappropriate content in code or comments, **the responsibility falls on you, the contributor**.

   - **You are expected to:**
     1. Review all AI-generated output before committing
     2. Remove any inappropriate, hostile, or threatening content
     3. Do NOT commit AI-generated death threats or hostile language under any circumstances
     4. If the AI misbehaves, fix it yourself or use a different prompt — do not blame the tool

   - "The AI made me do it" is **not an acceptable defense**. You are the human in the loop. You are responsible for what you commit.
