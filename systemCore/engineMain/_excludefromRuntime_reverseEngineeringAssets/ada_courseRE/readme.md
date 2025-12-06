
# Ada on macOS, Linux, & Windows: A Developer's Survival Guide

This document serves as a personal field manual for developing robust, portable Ada applications using the Alire (`alr`) and GNAT toolchain. It is the result of extensive debugging and reverse-engineering of the build process on macOS and provides a clear, reliable workflow for all platforms.

## 1. Core Philosophy: Robustness Through Portability

After encountering numerous issues with toolchain configurations and library compatibility, the guiding principle for all future projects is: **prioritize maximum portability and robustness.**

This means:
*   **Favor Standard Libraries:** Use libraries defined in the Ada Language Standard (`Ada.*`, `Interfaces.*`) as the first choice.
*   **Avoid GNAT-Specific Libraries:** While convenient, libraries like `GNAT.OS_Lib` are non-standard and have demonstrated inconsistent behavior or availability across different toolchain versions or platforms. They are a source of portability risk and should be avoided for critical code.
*   **Embrace C Bindings for OS Interaction:** For low-level tasks, the most reliable method is to bind directly to the underlying C library functions of the operating system. This gives ultimate control and bypasses any abstractions that might be incomplete or buggy.

## 2. The Role of `Interfaces.C`: The "Manual Transmission"

The decision to use C bindings is a powerful but deliberate trade-off. It should be reserved for the **1%** of the code that absolutely must interface with the raw operating system. The other **99%** of the application logic should remain in pure, safe Ada.

Using `Interfaces.C` is **exactly like switching a car from an automatic to a manual transmission.**

| Feature | **Pure Ada (Automatic)** | **Ada with `Interfaces.C` (Manual)** |
| :--- | :--- | :--- |
| **How it Works** | The compiler and runtime handle dangerous details for you automatically. | You are given direct, low-level control and are responsible for every step. |
| **Memory** | **Automatic.** Memory is managed for you (RAII). The car shifts gears for you. | **Manual.** You are responsible for every `malloc`/`free` or `New_String`/`Free`. You are shifting the gears yourself. |
| **Safety** | **High.** The language prevents common errors like null pointer access and buffer overflows at compile-time or with runtime checks. | **None.** The safety net is removed at the C boundary. Mistakes lead to crashes. The system assumes the driver is an expert. |
| **Result of a Mistake** | A compile-time error or a handleable runtime exception. | A `Segmentation fault` (segfault) or other undefined behavior. |

**Conclusion on Reliability:**
This approach is used to create a small, well-defined, and intensely scrutinized "wrapper" package in your Ada code. You build a tiny, unsafe bridge to C, and then the rest of your large, safe Ada application can use that bridge without ever having to touch C directly again, thus preserving the overall safety of the system.

## 3. Case Study: The "Not Declared in OS_Lib" Mystery

This is one of the most common and confusing errors for newcomers to GNAT.

**The Symptom:**
A new project fails to compile with a large wall of errors like `"No_Process_Id" not declared in "OS_Lib"` and `missing argument for parameter "Output_File" in call to "Spawn" declared at s-os_lib.ads`.

**Diagnosis:**
The compiler is not using the intended user-facing `GNAT.OS_Lib` library. Instead, it has fallen back to an internal, low-level package `System.OS_Lib` (found in `s-os_lib.ads`). This happens because the source code is missing the correct `with` clause.

**The Solution:**
The fix is to be explicit: `with GNAT.OS_Lib;`. However, given the portability issues discovered, the **recommended solution is to avoid `GNAT.OS_Lib` entirely** and use a C binding instead.

> **Warning!**
> Be cautious when using `GNAT.OS_Lib`. Experience has shown this non-standard library can cause significant build and portability issues.

## 4. The Alire Environment: Why `alr exec --` is Essential

A fundamental concept of Alire is that it creates a managed, self-contained environment for each project. It does **not** install compilers or libraries globally. This is why we must use the `alr exec --` command.

The command `alr exec -- <command>` acts as a "gatekeeper" that reads the project's manifest and constructs a temporary, perfectly configured environment for the command to run in. This ensures the correct GNAT version and all third-party dependencies are found.

**The Golden Rule:** Any tool that needs to interact with your project's specific compiler or dependencies **MUST be run via `alr exec --`**.

## 5. Case Study: Alire Dependency and Toolchain Hell

Attempting to use third-party libraries like `ashell` can fail if the toolchain is not correctly configured.

**The Symptom:** `alr with ashell` failed with `Missing: +❗ gcc ^11.2.4`.

**Diagnosis:** The library declared a dependency on a GNAT/GCC compiler, but Alire was checking the system's default `gcc` (Apple `clang`) and failing.

**The Fix:**
1. Run `alr toolchain --select`.
2. From the interactive menu, choose the `gnat_native` option, **not** "None" or "system".

## 6. Project Build Workflow: The "zephy" StellaIcarus Daemon Module

This section provides a complete, step-by-step workflow for initializing, building, and running a new Ada project, using the "zephy" module as an example. This process synthesizes all the lessons learned in this document.

### Step 1: Project Initialization

```bash
alr -n init --bin zephy
cd zephy
```

### Step 2: Adding Dependencies (If Necessary)

```bash
alr with <library_name>
alr update
```

### Step 3: Writing the Application Code

Update `zephy.gpr` to point to your main file: `for Main use ("zephy_daemon.adb");`.

### Step 4: Building the Executable

**For macOS:**
```bash
alr exec -- gprbuild -largs -L"$(xcrun --show-sdk-path)/usr/lib"
```

**For Linux & Windows:**
```bash
alr exec -- gprbuild
```

### Step 5: Running the Daemon

```bash
./bin/zephy_daemon
```

## 7. The "Ghost in the Machine" Protocol: The Ultimate Reset

When the build system behaves illogically, the environment is irrecoverably corrupt.
1.  **Back up and remove the Alire cache:** `mv ~/.alire ~/.alire_BACKUP`
2.  **Clean the shell config:** Edit `~/.zshrc` or similar and remove the Alire `PATH` entry.
3.  **Restart the terminal completely.**
4.  **Reinstall Alire from scratch.**
5.  **Restart the terminal again.**

---
## 8. Workflow for a Provable SPARK Project: Hello Sekai

This section details the workflow for creating a high-integrity application using SPARK, a subset of Ada designed for formal verification. We will mathematically prove our code is free from certain classes of runtime errors. This process is the result of debugging the toolchain and discovering the most robust method for creating a self-contained, provable project.

### Step 1: Project Initialization

Create a new binary crate for our SPARK project.

```bash
alr init --bin hello_sekai_spark
cd hello_sekai_spark
```

### Step 2: Add the `gnatprove` Tool Dependency

By default, a new project does not have access to the `gnatprove` formal analysis engine. Attempts to run it will fail with `Executable not found in PATH`.

While one could use `alr toolchain --select` to pick a global SPARK-enabled compiler, a more robust and portable method is to declare the prover as a **project-specific tool dependency.**

> **The Self-Contained Project Principle:**
> By adding `gnatprove` with `alr with`, the project's manifest (`alire.toml`) now explicitly records the tools it needs. This makes the project immune to global toolchain configuration changes and guarantees that any developer can check it out and build it with a single command.

Run the following command to add the prover to your project:

```bash
alr with gnatprove
```

### Step 3: Configure the Project File for Proof

We must tell the build system that we intend to prove this project. Open `hello_sekai_spark.gpr` and add the `package Prove`.

```gpr
project Hello_Sekai_Spark is
   for Source_Dirs use ("src");
   for Object_Dir use "obj";
   for Main use ("hello_sekai_spark.adb");

   -- Add this entire section to enable formal proof
   package Prove is
      for Proof_Switches ("Ada") use ("--level=0");
   end Prove;

end Hello_Sekai_Spark;
```

### Step 4: The Core SPARK Pattern: Separate Logic from I/O

The single most important concept in SPARK is the separation of provable logic from unprovable side-effects (like console I/O). The prover cannot reason about `Ada.Text_IO`, so we must isolate it.

The pattern is:
1.  **The Provable Core:** A pure SPARK package with contracts that performs calculations and returns results.
2.  **The Unprovable Shell:** A standard Ada procedure (marked with `SPARK_Mode => Off`) that calls the provable core and handles all I/O.

**1. Create the Provable Core (`src/message_generator.ads` and `.adb`)**

*   `src/message_generator.ads` (The Specification with a Contract)
    ```ada
    package Message_Generator
      with SPARK_Mode -- This package must be valid SPARK code
    is
       function Get_Message return String
         with Post => Get_Message'Result = "Hello, Sekai! (Proven and Separated)";
    end Message_Generator;
    ```
*   `src/message_generator.adb` (The Implementation)
    ```ada
    package body Message_Generator
      with SPARK_Mode
    is
       function Get_Message return String is
       begin
          return "Hello, Sekai! (Proven and Separated)";
       end Get_Message;
    end Message_Generator;
    ```

**2. Create the Unprovable Shell (`src/hello_sekai_spark.adb`)**

Replace the default main file with this code, which acts as the harness.

```ada
with Ada.Text_IO;
with Message_Generator; -- Import our provable package

procedure Hello_Sekai_Spark
  with SPARK_Mode => Off -- Explicitly tell the prover to ignore this file
is
   The_Message : constant String := Message_Generator.Get_Message;
begin
   Ada.Text_IO.Put_Line (The_Message);
end Hello_Sekai_Spark;
```

### Step 5: Prove, Build, and Run

The three-stage command sequence is now:

1.  **Prove the Logic:** Run `gnatprove`. It will analyze `message_generator` and ignore `hello_sekai_spark`.
    ```bash
    alr exec -- gnatprove -P hello_sekai_spark.gpr
    ```

2.  **Build the Executable:** Use `gprbuild` as usual.
    ```bash
    # For macOS
    alr exec -- gprbuild -P hello_sekai_spark.gpr -largs -L"$(xcrun --show-sdk-path)/usr/lib"

    # For Linux & Windows
    alr exec -- gprbuild -P hello_sekai_spark.gpr
    ```

3.  **Run the Final Program:**
    ```bash
    ./bin/hello_sekai_spark
    ```
    You will see the output: `Hello, Sekai! (Proven and Separated)`