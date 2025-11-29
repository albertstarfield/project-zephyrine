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
        
    def run_qrnn(self, feature_sequence: list[int], priority_lock: 'PriorityQuotaLock') -> np.ndarray:
        """
        Runs the QRNN simulation on a sequence of 16-bit features.

        :param feature_sequence: A list of 16-bit integers representing historical interaction features.
        :param priority_lock: The priority lock to use for preemption.
        :return: A predicted 1024-dimensional vector.
        """
        if not self.backend:
            logger.error("QRNN simulation skipped: backend not initialized.")
            return np.zeros(1024)

        if not feature_sequence:
            logger.warning("QRNN simulation skipped: empty feature sequence.")
            return np.zeros(1024)

        logger.debug(f"Running QRNN simulation with {self.backend} backend on a sequence of {len(feature_sequence)} features.")

        # Acquire lock
        if priority_lock:
            priority_lock.acquire(0)

        try:
            if self.backend in ["qiskit_qrack", "qiskit_aer"]:
                # --- Qiskit QRNN Logic ---
                num_qubits = 16
                qc = QuantumCircuit(num_qubits, num_qubits)

                for i, feature in enumerate(feature_sequence):
                    if priority_lock and priority_lock.is_preempted(0):
                        raise Exception("Task preempted by higher priority task.")
                    
                    for j in range(num_qubits):
                        if (feature >> j) & 1:
                            angle = float(feature) / float(2**16) * np.pi
                            qc.ry(angle, j)
                    
                    for j in range(num_qubits - 1):
                        qc.cnot(j, j + 1)

                # Get the state vector from the simulation
                qc.save_statevector()
                compiled_circuit = transpile(qc, self.qiskit_simulator)
                job = self.qiskit_simulator.run(compiled_circuit)
                result = job.result()
                state_vector = result.get_statevector(compiled_circuit)

                # Convert state vector to a 1024-dimensional vector
                real_vector = np.abs(state_vector)
                reshaped_vector = real_vector.reshape((256, 256))
                downsampled_vector = np.mean(reshaped_vector, axis=0)
                
                # Pad to 1024 dimensions
                predicted_vector = np.zeros(1024)
                predicted_vector[:256] = downsampled_vector
                
                logger.debug(f"  Qiskit simulation complete. Returning predicted vector.")
                return predicted_vector

            # --- Numpy QRNN Logic ---
            num_qubits = 16
            state_vector = np.zeros(2**num_qubits, dtype=np.complex64)
            state_vector[0] = 1

            for i, feature in enumerate(feature_sequence):
                if priority_lock and priority_lock.is_preempted(0):
                    raise Exception("Task preempted by higher priority task.")

                for j in range(num_qubits):
                    if (feature >> j) & 1:
                        # Apply RY gate with an angle based on the feature
                        angle = float(feature) / float(2**16) * np.pi
                        ry = self._ry_gate(angle)
                        gate = np.identity(2**(j), dtype=np.complex64)
                        gate = np.kron(ry, gate)
                        gate = np.kron(np.identity(2**(num_qubits - j - 1), dtype=np.complex64), gate)
                        state_vector = np.dot(gate, state_vector)

                # Entangle qubits
                for j in range(num_qubits - 1):
                    cnot = self._cnot_gate(num_qubits, j, j + 1)
                    state_vector = np.dot(cnot, state_vector)

            # Simulate depolarizing noise
            noise_level = 0.01
            for i in range(num_qubits):
                if np.random.rand() < noise_level:
                    x = self._x_gate()
                    gate = np.identity(2**(i), dtype=np.complex64)
                    gate = np.kron(x, gate)
                    gate = np.kron(np.identity(2**(num_qubits - i - 1), dtype=np.complex64), gate)
                    state_vector = np.dot(gate, state_vector)

            # Convert the final state_vector to a 1024-dimensional real-valued vector
            real_vector = np.abs(state_vector)
            reshaped_vector = real_vector.reshape((256, 256))
            downsampled_vector = np.mean(reshaped_vector, axis=0)
            
            # Pad to 1024 dimensions
            predicted_vector = np.zeros(1024)
            predicted_vector[:256] = downsampled_vector

            logger.debug(f"  QRNN simulation complete. Returning predicted vector.")

            return predicted_vector
        finally:
            if priority_lock:
                priority_lock.release(0)

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