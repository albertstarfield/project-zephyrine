import numpy as np
from loguru import logger
import os
import sys
import subprocess
import pickle
import tempfile
import argparse

# Check for Qiskit availability (only needed for the main provider, not the worker)
try:
    from qiskit import QuantumCircuit, transpile, Aer
    from qiskit.providers.qrack import Qrack

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


# --- ISOLATED MATH FUNCTIONS (Static/Standalone) ---
# These are moved outside the class so the worker can use them without overhead.

def _x_gate():
    return np.array([[0, 1], [1, 0]], dtype=np.complex64)


def _ry_gate(angle):
    cos = np.cos(angle / 2)
    sin = np.sin(angle / 2)
    return np.array([[cos, -sin], [sin, cos]], dtype=np.complex64)


def _cnot_gate(num_qubits, control, target):
    # WARNING: This creates dense matrices. For >14 qubits, this is extremely memory heavy.
    # Ideally, this should use sparse matrices, but we preserve original logic for consistency.
    dim = 2 ** num_qubits
    gate = np.identity(dim, dtype=np.complex64)

    # Pre-calculate indices to speed up loop
    for i in range(dim):
        # Bit manipulation to check control bit
        if (i >> control) & 1:
            # Flip target bit
            j = i ^ (1 << target)

            # Apply swap in matrix
            gate[i, i] = 0
            gate[j, j] = 0
            gate[i, j] = 1
            gate[j, i] = 1
    return gate


def _execute_numpy_qrnn_isolated(state_sequence: list[np.ndarray]) -> np.ndarray:
    """
    The heavy lifting logic, isolated from the class.
    """
    if not state_sequence:
        return np.zeros(1024)

    num_qubits = len(state_sequence[0])
    # Initialize state |00...0>
    state_vector = np.zeros(2 ** num_qubits, dtype=np.complex64)
    state_vector[0] = 1.0

    for t, input_features in enumerate(state_sequence):
        # 1. Encoding Layer (Angle Encoding)
        for j in range(num_qubits):
            theta = input_features[j] * np.pi
            ry = _ry_gate(theta)

            # Tensor Product expansion
            # Note: Optimized slightly to avoid recursion if possible, but keeping kron for logic match
            gate = np.identity(2 ** (j), dtype=np.complex64)
            gate = np.kron(ry, gate)
            gate = np.kron(np.identity(2 ** (num_qubits - j - 1), dtype=np.complex64), gate)

            state_vector = np.dot(gate, state_vector)

        # 2. Entanglement Layer (CNOT chain)
        for j in range(num_qubits - 1):
            cnot = _cnot_gate(num_qubits, j, j + 1)
            state_vector = np.dot(cnot, state_vector)

        # Circular entanglement
        cnot_last = _cnot_gate(num_qubits, num_qubits - 1, 0)
        state_vector = np.dot(cnot_last, state_vector)

    # --- Measurement / Readout ---
    probabilities = np.abs(state_vector) ** 2

    # Map 2^16 (65536) -> 1024
    target_dim = 1024
    if len(probabilities) >= target_dim:
        chunk_size = len(probabilities) // target_dim
        reshaped = probabilities[:chunk_size * target_dim].reshape((target_dim, chunk_size))
        predicted_vector = np.mean(reshaped, axis=1)
    else:
        # Pad if too small (unlikely for 16 qubits)
        predicted_vector = np.pad(probabilities, (0, target_dim - len(probabilities)))

    # Normalize
    norm = np.linalg.norm(predicted_vector)
    if norm > 0:
        predicted_vector = predicted_vector / norm

    return predicted_vector.astype(np.float32)


class VectorComputeProvider:
    """
    Orchestrator for Quantum Recurrent Neural Network (QRNN) simulation.
    Manages backends and process isolation.
    """

    def __init__(self, use_gpu: bool = True, priority_lock: 'PriorityQuotaLock' = None):
        self.backend = None
        self.qiskit_simulator = None
        self.priority_lock = priority_lock
        self.initialize_backend(use_gpu)

    def initialize_backend(self, use_gpu: bool):
        # (Existing backend selection logic kept for Qiskit/QPU support)
        logger.info("Initializing Vector Compute Provider...")
        qpu_provider = os.environ.get("QPU_PROVIDER", "").lower()

        if qpu_provider == "qrack" and QISKIT_AVAILABLE:
            try:
                self.qiskit_simulator = Qrack.get_backend('qasm_simulator')
                self.backend = "qiskit_qrack"
                logger.info("✅ Qiskit-Qrack simulator initialized.")
                return
            except Exception:
                pass

        if qpu_provider == "qiskit" and QISKIT_AVAILABLE:
            try:
                self.qiskit_simulator = Aer.get_backend('aer_simulator')
                self.backend = "qiskit_aer"
                logger.info("✅ Qiskit Aer simulator initialized.")
                return
            except Exception:
                pass

        # Default fallback to the subprocess-isolated numpy engine
        self.backend = "subprocess_numpy"
        logger.info("Using Subprocess-Isolated Numpy for quantum simulation (Memory Safe).")

    def run_qrnn(self, state_sequence: list[np.ndarray], priority_lock: 'PriorityQuotaLock') -> np.ndarray:
        """
        Runs the QRNN. If using the numpy backend, it launches a separate process
        to ensure memory is completely flushed after the massive matrix operations.
        """
        if not self.backend:
            logger.error("QRNN skipped: backend not initialized.")
            return np.zeros(1024)

        # Handle Qiskit/Other backends directly (assuming they manage memory better or we want speed)
        if self.backend.startswith("qiskit"):
            # Placeholder for Qiskit implementation logic...
            # For now, fallback to numpy logic if implementation is missing
            pass

            # --- SUBPROCESS ISOLATION LOGIC ---
        # This guarantees memory reclamation.
        if priority_lock:
            priority_lock.acquire(0)
            # Registering the process happens inside the subprocess block below

        temp_input = None
        temp_output = None

        try:
            # 1. Serialize Input
            # We use a temp file to pass the large arrays efficiently
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
                pickle.dump(state_sequence, f)
                temp_input = f.name

            temp_output = temp_input + ".out"

            # 2. Prepare Command
            # We call this same file as a script
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--worker",
                   "--input", temp_input,
                   "--output", temp_output]

            # 3. Execution
            logger.debug(f"QRNN: Launching isolated worker process for {len(state_sequence)} steps...")

            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # If we have a lock, register the PID so the Watchdog/Lock manager can kill it if needed
            if priority_lock and hasattr(priority_lock, "set_holder_process"):
                priority_lock.set_holder_process(process)

            # Wait for completion
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                logger.error(f"QRNN Worker failed (RC={process.returncode}): {stderr}")
                return np.zeros(1024)

            # 4. Deserialize Output
            if os.path.exists(temp_output):
                with open(temp_output, "rb") as f:
                    result = pickle.load(f)
                return result
            else:
                logger.error("QRNN Worker did not produce output file.")
                return np.zeros(1024)

        except Exception as e:
            logger.error(f"QRNN Execution Error: {e}")
            return np.zeros(1024)

        finally:
            if priority_lock:
                priority_lock.release()

            # Cleanup temp files
            for f in [temp_input, temp_output]:
                if f and os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass


# ==========================================
# WORKER ENTRY POINT
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help="Run in worker mode")
    parser.add_argument("--input", type=str, help="Input pickle file")
    parser.add_argument("--output", type=str, help="Output pickle file")

    args, unknown = parser.parse_known_args()

    if args.worker:
        # --- WORKER MODE ---
        try:
            # 1. Load Input
            with open(args.input, "rb") as f:
                seq = pickle.load(f)

            # 2. Run Heavy Computation
            # This allocates massive memory for matrices
            result_vector = _execute_numpy_qrnn_isolated(seq)

            # 3. Save Output
            with open(args.output, "wb") as f:
                pickle.dump(result_vector, f)

            # 4. Exit
            # OS cleans up all memory immediately
            sys.exit(0)

        except Exception as e:
            sys.stderr.write(f"Worker Exception: {str(e)}\n")
            sys.exit(1)

    else:
        # --- TEST MODE ---
        logger.info("--- VectorComputeProvider Self-Test (Main) ---")


        # Dummy Lock for testing
        class DummyLock:
            def acquire(self, p): pass

            def release(self, p): pass

            def set_holder_process(self, p): pass


        vcp = VectorComputeProvider(priority_lock=DummyLock())

        # Test Data (16 dimensions for 16 qubits)
        test_seq = [np.random.rand(16).astype(np.float32) for _ in range(5)]

        logger.info("Running simulation...")
        res = vcp.run_qrnn(test_seq, DummyLock())
        logger.success(f"Result Vector Shape: {res.shape}")
        logger.info(f"Result Sample: {res[:5]}")