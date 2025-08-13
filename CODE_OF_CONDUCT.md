# Contributing to Project Zephyrine

Thank you for your interest in contributing! Adherence to these guidelines is essential for maintaining the quality, integrity, and traceability of the project.


# Contributor Covenant Code of Conduct

All contributors are expected to read and abide by our [Code of Conduct](./CODE_OF_CONDUCT.md). All interactions within this project are governed by it.

## Article I: Pledge

1.1. We, the members, contributors, and leaders of this community, solemnly pledge to create an environment of unwavering inclusivity and freedom from harassment for all participants, irrespective of their age, physical attributes, visible or concealed disabilities, racial or ethnic background, gender identity and expression, experience level, educational background, socioeconomic status, nationality, personal appearance, race, religion, sexual orientation, or any other distinguishing characteristic.

1.2. We commit to engaging with one another in a manner that fosters an atmosphere of openness, receptiveness, diversity, inclusivity, and well-being.

## Article II: Standards

2.1. We consider the following behaviors conducive to maintaining a positive community environment:

   - Demonstrating empathy and kindness towards fellow community members.
   - Respecting divergent opinions, viewpoints, and life experiences.
   - Offering constructive feedback gracefully and receiving it with humility.
   - Taking accountability for our actions, extending apologies to those adversely affected by our errors, and learning from such experiences.
   - Prioritizing the collective welfare of the community over individual interests.
   - **Adhering to the project's architectural principles and development workflow. This includes, but is not limited to, refraining from committing compiled, architecture-specific binaries. The `launcher.py` script is the sole authority for local compilation to ensure portability and reproducibility for all contributors.**

2.2. Behaviors that fall below the accepted standards include:

   - Employing sexualized language or imagery and making sexual advances in any form.
   - Engaging in trolling, issuing insults or derogatory comments, and launching personal or political attacks.
   - Perpetrating public or private harassment.
   - Disseminating others' private information, such as physical addresses or email addresses, without explicit consent.
   - Conduct that could reasonably be viewed as inappropriate within a professional setting.

## Development Principles

### The Launcher is the Source of Truth

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

All commit messages must follow the **Formal Traceability Standard**. This is non-negotiable and is required for all merges.

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
    -   `build`: Changes that affect the build system or external dependencies (e.g., `launcher.py` modifications, `go.mod` updates).

-   **`scope`**: The component or module affected by the change (e.g., `mesh`, `watchdog`, `launcher`, `tui`).

-   **`subject`**: A concise, imperative-mood description of the change.
    -   Use the present tense ("add feature" not "added feature").
    -   Do not capitalize the first letter.
    -   Do not end with a period.

**Example Title:** `fix(mesh): correct asset path resolution in manifest generation`

#### **2. Body (Optional but Recommended)**

The body provides the context and "why" of the change. Explain the problem, the reasoning behind the solution, and any trade-offs made.

#### **3. Footer (Mandatory for Features and Fixes)**

The footer is used for explicit traceability.

-   It **MUST** reference the ID(s) from the `Issue_Log.md` document.
-   Use keywords like `Resolves:`, `Fixes:`, or `Implements:`.

**Example Footer:** `Resolves: ZM-BUG-004`

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

## Article III: Enforcement Responsibilities

3.1. Community leaders bear the responsibility of elucidating and enforcing our criteria for acceptable conduct. They shall respond judiciously and fairly to any actions deemed inappropriate, menacing, offensive, or detrimental to the community.

3.2. Community leaders possess the authority and obligation to eliminate, amend, or dismiss comments, code contributions, wiki edits, issues, and other contributions that do not adhere to this Code of Conduct. They shall communicate the rationale behind moderation decisions when deemed appropriate.

## Article IV: Scope

4.1. This Code of Conduct is applicable in all community domains and extends to situations in which an individual officially represents the community in public settings. Examples of such representation encompass the use of official email addresses, postings through sanctioned social media accounts, or acting as an appointed envoy at online or offline gatherings.

## Article V: Enforcement Procedures

5.1. Instances of abusive, harassing, or otherwise inappropriate conduct should be brought to the attention of Albert Starfield Wahyu Suryo Samudro and/or William Santoso through email at albertstarfield2001[at]gmail.com or willy030125[at]gmail.com, or through any accessible means. All complaints will be subjected to swift and equitable review and investigation.

5.2. All community leaders are obligated to safeguard the confidentiality and security of the complainant in any incident.

## Article VI: Enforcement Guidelines

6.1. Community leaders will adhere to the Community Impact Guidelines outlined below when determining consequences for actions violating this Code of Conduct:

### Section 1: Correction

   - **Community Impact**: Use of inappropriate language or any behavior considered unprofessional or unwelcome within the community.
   - **Consequence**: A private, written warning from community leaders, providing clarification regarding the nature of the violation and an explanation for the inappropriate behavior. A public apology may be requested.

### Section 2: Warning

   - **Community Impact**: A single violation or a series of actions.
   - **Consequence**: A warning with repercussions for continued misconduct. No interaction with the involved parties, including unsolicited communication with those enforcing the Code of Conduct, is permitted for a specified duration. This encompasses refraining from interactions within community spaces and external channels such as social media. Violation of these terms may result in a temporary or permanent ban.

### Section 3: Temporary Ban

   - **Community Impact**: A severe breach of community standards, including prolonged inappropriate behavior.
   - **Consequence**: A temporary suspension from any form of interaction or public communication with the community for a specified period. During this period, no public or private interaction with the involved parties, including unsolicited interaction with those enforcing the Code of Conduct, is allowed. Violating these terms may lead to a permanent ban.

### Section 4: Permanent Ban

   - **Community Impact**: Demonstrating a recurring pattern of community standards violations, including persistent inappropriate conduct, harassment of an individual, or hostility or denigration towards specific groups.
   - **Consequence**: A permanent ban from all forms of public interaction within the community.

## Article VII: Attribution

7.1. This Code of Conduct is adapted from the [Contributor Covenant][homepage], version 2.0, which can be accessed at [https://www.contributor-covenant.org/version/2/0/code_of_conduct.html](https://www.contributor-covenant.org/version/2/0/code_of_conduct.html).

7.2. The Community Impact Guidelines draw inspiration from [Mozilla's code of conduct enforcement ladder](https://github.com/mozilla/diversity).

[homepage]: https://www.contributor-covenant.org

For answers to frequently asked questions about this code of conduct, please consult the FAQ at [https://www.contributor-covenant.org/faq](https://www.contributor-covenant.org/faq). Translations are available at [https://www.contributor-covenant.org/translations](https://www.contributor-covenant.org/translations).
