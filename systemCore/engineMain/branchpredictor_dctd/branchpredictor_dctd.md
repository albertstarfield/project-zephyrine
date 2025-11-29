# Dancing in the Celestial Time-stream (DCTD) - Implementation Plan & Architecture

**Project:** `zephy`
**Module:** `systemCore/engineMain`
**Target Modules:** `VectorCompute_Provider.py`, `dctd_branchpredictor.py`

## 1. Architectural Context & Logic Note (Python Implementation)

### The "Negative Runtime" Branch Prediction System
This subsystem implements a Class B, quantum-enhanced branch predictor to achieve "negative runtime" for high-priority (`ELP1`) generation tasks. By utilizing a low-priority (`ELP0`) background task, the system predicts the probability distribution of future vector states based on the historical "rhythm" of interactions.

The new architecture for the branch predictor is a pure Python implementation that is integrated directly into the main application. It consists of three main components:

1.  **`AdelaideAlbertCortex.py` (The Core Logic):**
    *   This is the main application logic that handles user interactions.
    *   It contains the "Temporal Sieve," which extracts recent interaction history from the database.
    *   It then passes this historical data to the `BranchPredictorProvider`.

2.  **`dctd_branchpredictor.py` (The Orchestrator):**
    *   This module acts as the orchestrator for the prediction process.
    *   It receives the historical data from `AdelaideAlbertCortex.py`.
    *   It then calls the `VectorCompute_Provider.py` to perform the actual quantum simulation.

3.  **`VectorCompute_Provider.py` (The Quantum Engine):**
    *   This module is responsible for simulating the Quantum Recurrent Neural Network (QRNN).
    *   It takes the historical data, encodes it into a quantum state, evolves the state, applies noise, and measures the result to get a prediction.
    *   This module is designed to use `pyqrack` for GPU-accelerated quantum simulation, with a `numpy`-based fallback for systems without a compatible GPU or `pyqrack` installation.

### How it Predicts the Next Context:

The prediction process works as follows:

1.  **Temporal Sieve:** When a new user request is received, the `AdelaideAlbertCortex.py` module first looks at the recent interaction history. It extracts a sequence of interaction vectors and their timestamps. This sequence represents the "rhythm" of the conversation.

2.  **Feature Extraction:** The sequence of interaction vectors is then processed to extract a sequence of 16-bit "features." Each feature is a combination of a 10-bit LSH (Locality Sensitive Hash) of the interaction vector and a 6-bit "time bin" representing the time elapsed since the previous interaction.

3.  **Quantum Simulation:** The sequence of 16-bit features is passed to the `dctd_branchpredictor.py`, which in turn passes it to the `VectorCompute_Provider.py`. The `VectorCompute_Provider.py` then performs the following steps:
    *   **Encoding:** The 16-bit features are encoded into the state of a 16-qubit quantum circuit.
    *   **Evolution:** The quantum state is "evolved" by applying a series of quantum gates (e.g., `CNOT` gates) to entangle the qubits and create correlations between them.
    *   **Noise Injection:** A small amount of "prophetic noise" is injected into the system. This allows the simulation to make creative, non-linear predictions.
    *   **Measurement:** All qubits are measured, and the result is a 16-bit integer that represents the predicted future state.

4.  **Prediction:** The 16-bit integer from the measurement is then processed to extract a 10-bit LSH hash. This hash is the predicted "content" of the next interaction.

5.  **Augmentation:** The `AdelaideAlbertCortex.py` module takes the predicted LSH hash and queries the database for past interactions that have a similar hash. The content of these past interactions is then injected into the user's prompt as "predicted future context." This allows the language model to generate a response with a better understanding of the likely topic of the next turn in the conversation, effectively giving it a "head start."

---

## 2. Implementation TODO List (Python)

### Phase 1: Database & Configuration (Completed)

*Goal: Prepare the data layer to support LSH hashing and configure system variables.*

1.  [x] **Update `database.py`:**
    *   Added `lsh_hash_10bit` (INTEGER) column to the `interactions` table.
    *   Implemented `calculate_and_store_lsh(vector)` using numpy.
2.  [x] **Update `CortexConfiguration.py`:**
    *   Added `DCTD_ENABLE_QUANTUM_PREDICTION`.

### Phase 2: The Quantum Engine (`VectorCompute_Provider.py`)

*Goal: Create a Python module that simulates the quantum computation.*

1.  [ ] **Create `VectorCompute_Provider.py`:**
    *   Implement a class `VectorComputeProvider`.
    *   **Hardware Switch Logic:**
        *   Implement a method to initialize a Qrack simulator (`pyqrack`).
        *   The method should have a fallback order: GPU (OpenCL) -> CPU.
    *   **QRNN Core Logic:**
        *   Implement a method `run_qrnn` that takes a sequence of 16-bit features.
        *   This method should implement the circuit generation: Encode Input -> Evolve State -> Apply Noise -> Measure.
        *   Return a predicted hash.

### Phase 3: The Branch Predictor Provider (`dctd_branchpredictor.py`)

*Goal: Create a provider that orchestrates the branch prediction process.*

1.  [ ] **Create `dctd_branchpredictor.py`:**
    *   Implement a class `BranchPredictorProvider`.
    *   This class will instantiate `VectorComputeProvider`.
    *   Implement a public method `predict_future_hash` that:
        *   Takes the temporal sieve data as input.
        *   Calls the `run_qrnn` method of the `VectorComputeProvider`.
        *   Returns the predicted hash and time horizon.

### Phase 4: Core Integration (`AdelaideAlbertCortex.py`)

*Goal: Inject the prediction logic into the background generation loop.*

1.  [ ] **Modify `AdelaideAlbertCortex.py`:**
    *   Import and instantiate `BranchPredictorProvider`.
    *   In `_direct_generate_logic`, replace the ZMQ client logic with calls to the new provider.
    *   Call `branch_predictor_provider.predict_future_hash` with the temporal sieve data.
    *   Use the returned prediction to query the database and augment the prompt.

### Phase 5: Orchestration (`launcher.py`)

*Goal: Clean up the old Ada-based daemon logic.*

1.  [ ] **Remove Ada Compilation Step:**
    *   Remove the call to `_compile_dctd_daemon` and the function itself.
2.  [ ] **Remove Daemon Launch Step:**
    *   Remove the call to `start_dctd_daemon_fast` and the function itself.
3.  [ ] **Remove Cleanup Logic:**
    *   Remove the DCTD daemon cleanup logic from `cleanup_processes`.
    *   Remove the `DCTD_DAEMON_DIR` and `DCTD_DAEMON_INSTALLED_FLAG_FILE` configurations.
