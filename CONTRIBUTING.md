## Development Principles

### The Launcher is the main startup sequence

The `launcher.py` script is the sole authority for installing dependencies and compiling all project components. This ensures that every contributor can build a functional, portable, and reproducible version of the application on their own machine from a clean source checkout.

**Committing compiled, architecture-specific binaries is strictly forbidden.** This is a core architectural principle. See Article II of our Code of Conduct.


## Issue and Requirement Tracking

To ensure project resilience in cases where the `.git` history may be unavailable, all substantive changes must be linked to an issue or requirement ID.

-   **Canonical List:** The master list of all issues, requirements, and defect IDs is maintained in the document: `documentation/DeveloperDocumentation/Issue_Log.md`
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

-   **`scope`**: The component or module affected by the change (e.g., `mesh`, `watchdog`, `launcher`, `tui`).

-   **`subject`**: A concise, imperative-mood description of the change.
    -   Use the present tense ("add feature" not "added feature").
    -   Do not capitalize the first letter.
    -   Do not end with a period.

**Example Title:** `fix(mesh): correct asset path resolution in manifest generation`



#### **2. Body (Mandatory for Features, Fixes, and Refactors)**

The body provides the context and "why" of the change. Explain the problem, the reasoning behind the solution, and any trade-offs made.

**For contributions involving Generative AI**, the body **must** also include the addendum as specified in **Article II, Section 2.3.2**, containing the **Logical Flow Concept** and **Cited Progr
amming Manuals**.

#### **3. Footer (Mandatory)**

The footer is used for explicit traceability and verification. It **MUST** contain the following components in this order:

-   **Issue
- /Requirement ID(s):** Reference the ID(s) from the `Issue_Log.md` document. Use keywords like `Resolves:`, `Fixes:`, or `Implements:` .

-   **Code of Conduct Acknowledgment:** A mandatory checklist item.
    -   `Have I read the Code of Conduct?: Yes`


-   **Verification Canary:** A unique, date-based string to verify that these specific instructions have been read and followed.
    -   The base string is: `IdonotCondoneanyCorruptionorMisuseLyingddmmyyyy`
    -   Replace `ddmmyyyy` with the current Gregorian date (e.g., `13082025`).
    -   **If the day (`dd`) is an ODD number (1, 3, 5, etc.):** The entire string must be in **lowercase**.
    -   **If the day (`dd`) is an EVEN number (2, 4, 6, etc.):** The entire string must be in its **original mixed case** (`Idonot...`).

---

### **Complete Example (for a commit made on August 13th, 2025 - an ODD day)**

```
fix(mesh): correct asset path resolution and expand manifest scope

This commit resolves a critical path resolution bug in the ZephyMesh node and expands the asset manifest to inc
lude the Hugging Face cache, ensuring full P2P distribution capability.

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
idonotcondoneanycorruptionormisuselying13082025
```

### **Complete Example (for a commit made on August 14th, 2025 - an EVEN day)**

```
feat(fmc): add new trim command to servo hook

<...>

Resolves: FMC-FEAT-012
Have I read the Code of Conduct?: Yes
IdonotCondoneanyCorruptionorMisuseLying14082025
```
---

### **Complete Example**

```
fix(mesh): correct asset path resolution and expand manifest scope

This commit resolves a critical path resolution bug in the ZephyMesh node and expands the asset manifest to include the Hugging Face cache, ensuring full P2P distribution capability.

The Go node was being launched with an incorrect working directory, causing it to construct invalid paths when scanning for local assets. This has been corrected by setting the `cwd` to the project root.

The manifest scan has been updated to recursively include the `huggingface_cache` directory, enabling the P2P sharing of non-GGUF models and configurations.

Resolves: ZM-BUG-004
Implements: MESH-REQ-001.C
```

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

   - Several established open-source communities and foundations strictly prohibit contributions that are generated by or assisted by Artificial Intelligence in any capacity. This project respects the policies and sovereignty of those communities.

   - By submitting any content to Project Zephyrine that has been created, modified, or assisted by a Generative AI—even for tasks as minor as paraphrasing—you, the contributor, explicitly acknowledge and agree to the following consequence: your contribution may serve as grounds for Project Zephyrine to be **permanently banned and excluded** from participation in those communities.

   - This is not a possibility ("may be banned") but a direct consequence ("will be banned") where such policies are enforced. Communities with explicit anti-AI contribution policies include, but are not limited to:
     - The Gentoo Project [1]
     - The NetBSD Project [2]
     - The Asahi Linux Project [3]

   - This section serves as a formal notice. Continued contribution of any GenAI-assisted content constitutes your acknowledgment and acceptance of this outcome. The integrity of our contribution history is paramount, and we will not obscure the use of AI tools to circumvent the policies of other projects.