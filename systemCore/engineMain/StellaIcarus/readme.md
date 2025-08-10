# Stella Icarus Subsystem: Developer Guide for Deterministic AI Tool

## 1. Overview

The Stella Icarus subsystem is the deterministic core of the Adelaide/Zephy AI. It is not an optional add-on; it is the foundation that enables the AI to operate with the **predictability, reliability, and safety** required for high-stakes applications, particularly those analogous to modern aviation and avionics systems.

Its purpose is to enforce **determinism** in two critical domains: the *content* of a response and the *timing* of the system's operation. It achieves this through two distinct but complementary components:

1.  **Python Hooks:** Fast-reflex, JIT-compiled modules for immediate, pattern-based responses.
2.  **Ada Daemons:** High-reliability, continuously running background processes for sensory data streams.

## 2. The Core Philosophy: Determinism in Avionics

In standard conversational AI, responses can be creative and unpredictable. In an avionics or procedural environment, this is a critical failure. Stella Icarus is designed to solve this by imposing strict determinism where it matters most.

> ### Response Determinism: The Trained Pilot
> When a pilot executes a checklist or a standard operating procedure, the actions must be **accurate, repeatable, and procedurally correct, 1:1 with their training**. There is no room for creative interpretation.
>
> The **Python + Numba Hooks** act as this "Trained Pilot." They intercept specific commands and execute them using pre-defined, high-performance code, guaranteeing the correct output every single time.

> ### Temporal Determinism: The Watchtower & Watchdog
> An aircraft's core systems must operate on a predictable schedule. A flight control computer cannot simply "take a long time" to respond. Safety systems must constantly monitor the aircraft's state, providing a reliable "heartbeat."
>
> The **Ada 2022 Daemons** act as this "Watchtower." They are high-reliability, real-time processes designed to run continuously, providing a predictable stream of data and ensuring the system's core awareness is always active. They are the safety-critical layer.

## 3. Python Hooks (Response Determinism)

### Purpose
Python Hooks provide immediate, accurate, and procedurally correct responses. They are designed for tasks that are deterministic and require microsecond-level latency.

### Performance Mandate: JIT Compilation
All performance-critical hooks **must** use a Just-In-Time (JIT) compiler like **Numba**. Standard interpreted Python is not sufficient. Our `basic_math_hook.py` demonstrates this, achieving a **4-7 microsecond (µs)** response time.

### Use Case: Deterministic Machine Learning (e.g., OpenCV)
Beyond simple math, a hook is the ideal place to run a small, fast, deterministic ML model. For example, an OpenCV hook could be written to:
- **`PATTERN`**: Match a request for "read gauge `[gauge_name]` from panel image".
- **`handler`**: Receive an image, crop to the gauge's known coordinates, and use a simple, pre-trained OpenCV model or template matching to read the needle's position and return the exact value. This is a deterministic perception task, not a generative one.

### The Plugin Contract
For a Python file in the `./StellaIcarus/` directory to be loaded, it must provide:
1.  **`PATTERN`**: A compiled regular expression (`re.Pattern`) anchored to match the entire user input.
2.  **`handler(match, user_input, session_id)`**: A function that executes the logic and returns a `str` response or `None`.

### Gold Standard: `basic_math_hook.py`
All new Python hooks **must** be modeled on the robust, defensive design of `basic_math_hook.py`. Key features to replicate are:
- **Numba JIT Compilation:** For the core logic.
- **Graceful Fallback:** A pure Python version of the logic must be provided.
- **Startup Self-Test:** Proactively check if Numba is functional on module load.
- **Runtime Fallback:** The `handler` must catch any exception from the Numba call and fall back to the Python version.
- **Comprehensive Error Handling:** Return clear, user-friendly error messages for invalid inputs (e.g., division by zero).

---

## 4. Ada Daemons (Temporal Determinism & Safety)

### Purpose
Ada Daemons are high-reliability, continuously running background processes that provide the AI's sensory data stream, acting as a system watchdog or a Fly-By-Wire (FBW) telemetry source. The choice of **Ada 2022** reflects the need for safety and real-time performance.

### Architecture & Communication
- **Alire Projects:** Each daemon must be a valid Ada project with an `alire.toml` file, located in a subdirectory of `./StellaIcarus_Ada/`.
- **Protocol:** Daemons communicate by printing a single-line, valid **JSON string to `stdout`** on each loop iteration. The Python `StellaIcarusAdaDaemonManager` captures this output.

Refer to the `./StellaIcarus_Ada/basic_AdaDaemon_communication/` project for a canonical implementation.

---

## 5. The Bridge Architecture: FBW and IFCS

The two subsystems do not speak to each other directly. They are bridged by the AI core in a manner analogous to a modern aircraft's flight control system.

1.  **Ada Daemon (The Fly-By-Wire System):** A high-reliability, real-time system that continuously outputs raw, objective data (`pitch: 5.2, roll: -1.8`). It is the source of truth for the aircraft's state. It is not "intelligent."
2.  **AI Core / LLM (The Integrated Flight Control System):** The intelligent, higher-level system. It consumes the continuous data stream from the FBW and combines it with the pilot's command (the user's prompt) to make an informed decision.
3.  **Python Hook (The Checklist Procedure):** When the pilot issues a simple, procedural command ("*Gear down*"), the IFCS doesn't need to "think"; it executes a pre-programmed, deterministic sequence. This is a Python hook.

**Data Flow:** `Ada Daemon -> stdout -> Python Manager -> Data Queue -> AI Core Context -> LLM Decision`

---

## 6. Critical Developer Notes & Path Forward

### The Stability Paradox: Why Our Python is More Stable Than Our Ada
Theoretically, Ada should be the pinnacle of stability. In practice, our current implementation faces a critical issue:

**The Ada daemons are prone to segmentation faults after successful compilation (approx. 50% failure rate).**

A segfault is a catastrophic, unrecoverable memory error. This indicates a severe bug in the Ada code, likely from bypassing Ada's safety features with `Unchecked_` pragmas or incorrect C library bindings.

Conversely, the **Python + Numba hooks have proven to be exceptionally stable.** This is due to a deliberate, defensive design that anticipates and survives failure through graceful fallbacks and comprehensive error handling.

### Design Choice: Why No Hot-Reloading
The `StellaIcarusHookManager` scans for hooks **only at application startup**. This is a deliberate design choice to enforce **determinism**. By fixing the set of active hooks for the application's entire runtime, we guarantee stable, predictable, and secure behavior, which is paramount in an avionics-grade system.

### Path Forward
-   **For Python Hooks:** All new hooks **must** adhere to the robust, defensive design pattern of `basic_math_hook.py`. Prioritize stability, graceful fallbacks, and JIT compilation. New hooks are placed in the `./StellaIcarus/` directory.
-   **For Ada Daemons:** The highest priority is to **identify and fix the root cause of the segmentation faults.** Development must focus on debugging memory access and ensuring all code leverages Ada's built-in safety features. New daemons are placed in a subdirectory of `./StellaIcarus_Ada/`.

Until the Ada implementation can guarantee stability, the Python hooks represent the more reliable and production-ready component of the Stella Icarus subsystem.