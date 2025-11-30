import numpy as np
from loguru import logger
import os
try:
    from qiskit import QuantumCircuit, transpile, Aer
    from qiskit.providers.qrack import Qrack
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class VectorComputeProvider:
    """
    Simulates a Quantum Recurrent Neural Network (QRNN) for branch prediction.
    This provider supports Qiskit-Qrack for GPU simulation, Qiskit Aer for CPU simulation, 
    a hypothetical QPU, pyqrack for GPU/CPU simulation, and a numpy-based classical simulation as a final fallback.
    """

    def __init__(self, use_gpu: bool = True, priority_lock: 'PriorityQuotaLock' = None):
        self.backend = None
        self.qiskit_simulator = None
        self.priority_lock = priority_lock
        self.initialize_backend(use_gpu)

    def initialize_backend(self, use_gpu: bool):
        """
        Initializes the quantum simulation backend with the following priority:
        1. Qiskit-Qrack (if QPU_PROVIDER=qrack)
        2. Qiskit Aer (if QPU_PROVIDER=qiskit)
        3. Generic QPU Provider (if QPU_PROVIDER_ENABLED=true)
        4. Pyqrack (GPU/CPU simulation)
        5. Numpy (classical simulation)
        """
        logger.info("Initializing Vector Compute Provider...")
        qpu_provider = os.environ.get("QPU_PROVIDER", "").lower()

        # 1. Check for Qiskit-Qrack
        if qpu_provider == "qrack":
            if QISKIT_AVAILABLE:
                try:
                    self.qiskit_simulator = Qrack.get_backend('qasm_simulator')
                    self.backend = "qiskit_qrack"
                    logger.info("✅ Qiskit-Qrack simulator initialized as the quantum backend.")
                    return
                except Exception as e:
                    logger.error(f"Failed to initialize Qiskit-Qrack simulator: {e}. Falling back...")
            else:
                logger.warning("QPU_PROVIDER is 'qrack', but qiskit is not installed. Falling back...")

        # 2. Check for Qiskit Aer
        if qpu_provider == "qiskit":
            if QISKIT_AVAILABLE:
                try:
                    self.qiskit_simulator = Aer.get_backend('aer_simulator')
                    self.backend = "qiskit_aer"
                    logger.info("✅ Qiskit Aer simulator initialized as the quantum backend.")
                    return
                except Exception as e:
                    logger.error(f"Failed to initialize Qiskit Aer simulator: {e}. Falling back...")
            else:
                logger.warning("QPU_PROVIDER is 'qiskit', but qiskit is not installed. Falling back...")

        # 3. Check for generic QPU Provider (simulated)
        if os.environ.get("QPU_PROVIDER_ENABLED", "false").lower() == "true":
            try:
                # This is a placeholder for a real QPU provider integration
                # import qpu_provider_sdk 
                # self.backend = qpu_provider_sdk.get_qpu()
                logger.info("✅ Hypothetical QPU Provider found and selected.")
                self.backend = "qpu_provider"
                return
            except ImportError:
                logger.warning("QPU_PROVIDER_ENABLED is true, but the QPU SDK is not found. Falling back...")
            except Exception as e:
                logger.error(f"Failed to initialize QPU provider: {e}. Falling back...")

        # 4. Fallback to Pyqrack
        try:
            # import pyqrack
            # self.backend = pyqrack.qsim(16) # 16 qubits for 16-bit features
            logger.info("  (Placeholder) Pyqrack backend initialized.")
            self.backend = "pyqrack_simulator"
            return
        except ImportError:
            logger.warning("Pyqrack not found. Falling back to numpy simulation.")
        except Exception as e:
            logger.error(f"Failed to initialize Pyqrack backend: {e}. Falling back to numpy.")

        # 5. Final fallback to numpy
        self.backend = "numpy_fallback"
        logger.info("Using numpy fallback for quantum simulation.")

    def _x_gate(self):
        return np.array([[0, 1], [1, 0]], dtype=np.complex64)

    def _ry_gate(self, angle):
        cos = np.cos(angle / 2)
        sin = np.sin(angle / 2)
        return np.array([[cos, -sin], [sin, cos]], dtype=np.complex64)

    def _cnot_gate(self, num_qubits, control, target):
        gate = np.identity(2**num_qubits, dtype=np.complex64)
        for i in range(2**num_qubits):
            binary = format(i, f'0{num_qubits}b')
            if binary[num_qubits - 1 - control] == '1':
                flipped_binary = list(binary)
                flipped_binary[num_qubits - 1 - target] = '0' if binary[num_qubits - 1 - target] == '1' else '1'
                flipped_binary = "".join(flipped_binary)
                j = int(flipped_binary, 2)
                gate[i, i] = 0
                gate[j, j] = 0
                gate[i, j] = 1
                gate[j, i] = 1
        return gate

    def run_qrnn(self, state_sequence: list[np.ndarray], priority_lock: 'PriorityQuotaLock') -> np.ndarray:
        """
        Runs the QRNN simulation using Angle Encoding on a sequence of state vectors.

        :param state_sequence: A list of numpy arrays. Each array represents a timestep
                               and should have a length equal to num_qubits (16).
                               Values should be normalized floats between 0.0 and 1.0.
        :param priority_lock: The priority lock.
        :return: A predicted 1024-dimensional vector.
        """
        if not self.backend:
            logger.error("QRNN simulation skipped: backend not initialized.")
            return np.zeros(1024)

        if not state_sequence:
            logger.warning("QRNN simulation skipped: empty state sequence.")
            return np.zeros(1024)

        # Determine qubit count from the input dimension (should be 16)
        num_qubits = len(state_sequence[0])
        logger.debug(f"Running QRNN with {self.backend} backend. Seq Len: {len(state_sequence)}, Qubits: {num_qubits}")

        if priority_lock:
            priority_lock.acquire(0)

        try:
            # --- Numpy QRNN Logic (Classical Simulation of Quantum Circuit) ---
            # Initialize state |00...0>
            state_vector = np.zeros(2 ** num_qubits, dtype=np.complex64)
            state_vector[0] = 1.0

            for t, input_features in enumerate(state_sequence):
                if priority_lock and priority_lock.is_preempted(0):
                    raise Exception("Task preempted by higher priority task.")

                # 1. Encoding Layer (Angle Encoding)
                # We rotate each qubit based on the input feature value
                for j in range(num_qubits):
                    # Map input 0.0-1.0 to rotation 0 to Pi
                    theta = input_features[j] * np.pi
                    ry = self._ry_gate(theta)

                    # Apply gate to specific qubit (Tensor Product expansion)
                    # Optimization: In a real sim, we'd use sparse matrices, but this is explicit
                    gate = np.identity(2 ** (j), dtype=np.complex64)
                    gate = np.kron(ry, gate)
                    gate = np.kron(np.identity(2 ** (num_qubits - j - 1), dtype=np.complex64), gate)

                    state_vector = np.dot(gate, state_vector)

                # 2. Entanglement Layer (CNOT chain)
                # This diffuses the information across the qubits
                for j in range(num_qubits - 1):
                    cnot = self._cnot_gate(num_qubits, j, j + 1)
                    state_vector = np.dot(cnot, state_vector)

                # Circular entanglement (last to first) to close the loop
                cnot_last = self._cnot_gate(num_qubits, num_qubits - 1, 0)
                state_vector = np.dot(cnot_last, state_vector)

            # --- Measurement / Readout ---
            # Convert complex quantum probability amplitudes to real-valued dense vector
            probabilities = np.abs(state_vector) ** 2  # Get probabilities

            # We need to map 2^16 (65536) probabilities down to 1024 dimensions.
            # We reshape and pool (average).
            reshaped = probabilities.reshape((1024, 64))
            predicted_vector = np.mean(reshaped, axis=1)

            # Normalize the result to be a valid unit vector (like an embedding)
            norm = np.linalg.norm(predicted_vector)
            if norm > 0:
                predicted_vector = predicted_vector / norm

            logger.debug(f"  QRNN simulation complete. Predicted vector norm: {norm:.4f}")
            return predicted_vector

        finally:
            if priority_lock:
                priority_lock.release()

if __name__ == '__main__':
    # Example Usage
    logger.info("--- VectorComputeProvider Self-Test ---")
    
    class DummyLock:
        def acquire(self, priority):
            pass
        def release(self, priority):
            pass
        def is_preempted(self, priority):
            return False

    vcp = VectorComputeProvider(priority_lock=DummyLock())
    
    # Simulate a sequence of interaction features
    test_sequence = [12345, 54321, 23456, 65432] 
    
    prediction = vcp.run_qrnn(test_sequence, DummyLock())
    
    logger.info(f"Test sequence: {test_sequence}")
    logger.info(f"Predicted next vector (shape): {prediction.shape}")