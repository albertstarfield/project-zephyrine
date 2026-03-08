
# ==============================================================================
# MODULE: VectorCompute_provider.py (Snowball-Enaga Framework)
# DESCRIPTION: Standard Linear Algebra utility for time-series state prediction.
#              Provides a deterministic interface for aerospace GNC vectors.
#              Utilizes process-isolated tensor contraction for memory safety.
# DAL LEVEL: DAL C (Isolated Logic)
# MAINTAINER: Zephyrine Foundation
# ==============================================================================
import numpy as np
from loguru import logger
import os
import sys
import subprocess
import pickle
import tempfile
import argparse
import math

# --- ACCELERATION LIBRARIES ---
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from qiskit import QuantumCircuit, transpile
    # Qiskit Aer/Qrack imports happen lazily to avoid heavy load if unused
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

try:
    from priority_lock import PriorityQuotaLock, ELP0, ELP1
    logger.info("✅ Successfully imported PriorityQuotaLock, ELP0, ELP1.")
except ImportError as e:
    logger.error(f"❌ Failed to import from priority_lock.py: {e}")
    logger.warning("    Falling back to standard threading.Lock for priority lock (NO PRIORITY/QUOTA).")
    # Define fallbacks so the rest of the code doesn't crash immediately
    import threading
    PriorityQuotaLock = threading.Lock # type: ignore
    ELP0 = 0
    ELP1 = 1
    # You might want to sys.exit(1) here if priority locking is critical
    # sys.exit(1) # Commented out for testing

# ==========================================
# 1. QUANTUM CIRCUIT LOGIC (The "Source of Truth")
# ==========================================
def build_qrnn_circuit(state_sequence: list[np.ndarray]) -> 'QuantumCircuit':
    """
    Constructs the canonical Quantum Circuit for the QRNN.
    This is the standard definition that can run on any QPU.
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit is required to build the circuit structure.")

    num_qubits = len(state_sequence[0])
    qc = QuantumCircuit(num_qubits)

    # Initialize |0...0> is implicit in Qiskit

    for input_features in state_sequence:
        # 1. Encoding Layer (RY rotations)
        for j in range(num_qubits):
            theta = input_features[j] * np.pi
            qc.ry(theta, j)

        # 2. Entanglement Layer (CNOT chain)
        for j in range(num_qubits - 1):
            qc.cx(j, j + 1)

        # Circular entanglement
        qc.cx(num_qubits - 1, 0)

    # FIX: Removed 'qc.save_statevector()'
    # This caused the AttributeError. The 'statevector_simulator' backend
    # in run_qrnn will capture the final state automatically.
    return qc


# ==========================================
# 2. TENSOR SIMULATION ENGINE (GPU/CPU)
# ==========================================
# This implements the logic of the circuit above using linear algebra,
# supporting NumPy (CPU), PyTorch (MPS/ROCm/CUDA), and potentially CuPy.

def _get_tensor_backend():
    """
    Selects the best available tensor library and device.
    Returns: (backend_module, device_str)
    """
    # 1. PyTorch (Covers MPS, ROCm, CUDA via torch.compile/backends)
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            return torch, "cuda"
        elif torch.backends.mps.is_available():
            return torch, "mps"
        elif hasattr(torch, 'xpu') and torch.xpu.is_available(): # Intel oneAPI
            return torch, "xpu"
        # If torch CPU is the only option, we might prefer NumPy for simplicity, 
        # but Torch CPU is also very fast (AVX2/512). Let's stick with Torch if installed.
        return torch, "cpu"
    
    # 2. NumPy (Universal Fallback)
    return np, "cpu"

def _execute_tensor_simulation(state_sequence: list[np.ndarray], lib, device):
    """
    Executes the QRNN logic using the chosen tensor library (lib) on (device).
    Uses the MEMORY-OPTIMIZED Tensor Contraction approach.
    """
    # Helper to create tensors on device
    def to_tensor(data, dtype=None):
        if lib == torch:
            t = torch.tensor(data, device=device, dtype=torch.complex64) # Use complex64 for speed
            return t
        else: # NumPy
            return np.array(data, dtype=np.complex64)

    # Math Helpers (Framework Agnostic)
    def get_ry(angle):
        cos = lib.cos(angle / 2)
        sin = lib.sin(angle / 2)
        if lib == torch:
            return torch.tensor([[cos, -sin], [sin, cos]], device=device, dtype=torch.complex64)
        return np.array([[cos, -sin], [sin, cos]], dtype=np.complex64)

    # Execution
    num_qubits = len(state_sequence[0])
    
    # State Vector: (2^N)
    # We initialize on the device
    if lib == torch:
        state = torch.zeros(2**num_qubits, dtype=torch.complex64, device=device)
        state[0] = 1.0
    else:
        state = np.zeros(2**num_qubits, dtype=np.complex64)
        state[0] = 1.0

    for input_features in state_sequence:
        # Convert input to tensor if needed
        if lib == torch:
            feats = torch.tensor(input_features, device=device, dtype=torch.float32)
        else:
            feats = input_features

        # 1. Encoding
        for j in range(num_qubits):
            if lib == torch:
                theta = feats[j] * math.pi # Use math.pi for scalar mult
            else:
                theta = feats[j] * np.pi
                
            gate = get_ry(theta)
            
            # Tensor Reshape & Contract (The Optimization)
            n_high = 1 << j
            n_low = 1 << (num_qubits - 1 - j)
            
            # Reshape state to isolate target qubit
            state = state.reshape(n_high, 2, n_low)
            
            # Contract: 'ij, kjl -> kil'
            if lib == torch:
                # PyTorch einsum/tensordot
                # (2,2) x (H, 2, L) -> we want to multiply along dim 1 of state
                state = torch.tensordot(gate, state, dims=([1], [1])) # Result: (2, H, L)
                state = state.permute(1, 0, 2) # Move qubit axis back to middle: (H, 2, L)
            else:
                state = np.tensordot(gate, state, axes=([1], [1]))
                state = np.transpose(state, (1, 0, 2))
                
            state = state.flatten()

        # 2. Entanglement (Index Permutation - Very fast on GPU)
        dim = 2**num_qubits
        if lib == torch:
            indices = torch.arange(dim, device=device, dtype=torch.long)
        else:
            indices = np.arange(dim, dtype=np.int32)

        def apply_cnot_perm(curr_state, c, t):
            # Logic: MSB 0, LSB N-1 (matching the loop)
            # Control/Target bit positions from LSB side (since reshape logic was LSB-focused)
            # Actually, reshape logic above treated index 0 as outer/higher dim.
            # Let's align: j=0 is first iter, n_high=1 -> It is the MSB.
            
            c_pos = num_qubits - 1 - c
            t_pos = num_qubits - 1 - t
            
            # Mask
            if lib == torch:
                ctrl_mask = (indices >> c_pos) & 1 == 1
                flip_indices = indices ^ (1 << t_pos)
                new_indices = torch.where(ctrl_mask, flip_indices, indices)
                return curr_state[new_indices]
            else:
                ctrl_mask = (indices >> c_pos) & 1 == 1
                flip_indices = indices ^ (1 << t_pos)
                new_indices = np.where(ctrl_mask, flip_indices, indices)
                return curr_state[new_indices]

        for j in range(num_qubits - 1):
            state = apply_cnot_perm(state, j, j + 1)
        state = apply_cnot_perm(state, num_qubits - 1, 0)

    # --- Readout ---
    probs = lib.abs(state) ** 2
    
    # Pooling
    target_dim = 1024
    if len(probs) >= target_dim:
        chunk = len(probs) // target_dim
        reshaped = probs[:chunk*target_dim].reshape(target_dim, chunk)
        if lib == torch:
            predicted = torch.mean(reshaped, dim=1)
        else:
            predicted = np.mean(reshaped, axis=1)
    else:
        # Padding
        if lib == torch:
            predicted = torch.nn.functional.pad(probs, (0, target_dim - len(probs)))
        else:
            predicted = np.pad(probs, (0, target_dim - len(probs)))
            
    # Normalize & Return as CPU Numpy
    if lib == torch:
        norm = torch.linalg.norm(predicted)
        if norm > 0: predicted = predicted / norm
        return predicted.cpu().detach().numpy()
    else:
        norm = np.linalg.norm(predicted)
        if norm > 0: predicted = predicted / norm
        return predicted


# ==========================================
# 3. PROVIDER CLASS
# ==========================================
class VectorComputeProvider:
    def __init__(self, use_gpu: bool = True, priority_lock: 'PriorityQuotaLock' = None):
        self.backend_type = "auto" # Default
        self.priority_lock = priority_lock
        
        # Check env for override
        self.qpu_provider = os.environ.get("QPU_PROVIDER", "").lower()
        if self.qpu_provider in ["qiskit", "qrack"]:
            self.backend_type = "qiskit_simulator"
        else:
            # Detect fastest tensor backend
            lib, dev = _get_tensor_backend()
            self.tensor_lib = lib
            self.tensor_device = dev
            self.backend_type = "tensor_simulation"
            logger.info(f"VectorCompute: Using Tensor Backend: {lib.__name__} on {dev}")

    def run_qrnn(self, state_sequence: list[np.ndarray], priority_lock: 'PriorityQuotaLock') -> np.ndarray:
        # --- PATH 1: REAL QPU / SIMULATOR (via Qiskit) ---
        if self.backend_type == "qiskit_simulator" and QISKIT_AVAILABLE:
            try:
                # 1. Build the Circuit Object (Language of Qiskit)
                qc = build_qrnn_circuit(state_sequence)
                
                # 2. Select Backend
                if self.qpu_provider == "qrack":
                    from qiskit.providers.qrack import Qrack
                    backend = Qrack.get_backend('qasm_simulator')
                else:
                    from qiskit_aer import Aer
                    backend = Aer.get_backend('statevector_simulator')
                
                # 3. Execute
                # Transpile for the target backend
                t_qc = transpile(qc, backend)
                job = backend.run(t_qc)
                result = job.result()
                
                # 4. Extract Statevector
                sv = result.get_statevector(qc)
                probs = np.abs(sv) ** 2
                
                # ... (Pooling logic similar to above) ...
                # For brevity, reusing the numpy pooling logic on the result
                target_dim = 1024
                chunk = len(probs) // target_dim
                reshaped = probs[:chunk*target_dim].reshape(target_dim, chunk)
                pred = np.mean(reshaped, axis=1)
                return pred / np.linalg.norm(pred)

            except Exception as e:
                logger.error(f"Qiskit execution failed: {e}. Falling back to Tensor Engine.")
                # Fallthrough to tensor engine

        # --- PATH 2: ISOLATED TENSOR ENGINE (Process Safe) ---
        # We run the tensor engine in a subprocess to ensure GPU/CPU memory is flushed.
        
        if priority_lock:
            priority_lock.acquire(0)

        temp_input, temp_output = None, None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
                pickle.dump(state_sequence, f)
                temp_input = f.name
            temp_output = temp_input + ".out"

            cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--input", temp_input, "--output", temp_output]
            
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if priority_lock and hasattr(priority_lock, "set_holder_process"):
                priority_lock.set_holder_process(proc)

            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                logger.error(f"Tensor Worker Failed: {stderr}")
                return np.zeros(1024)

            if os.path.exists(temp_output):
                with open(temp_output, "rb") as f:
                    return pickle.load(f)
            return np.zeros(1024)

        except Exception as e:
            logger.error(f"Worker launch failed: {e}")
            return np.zeros(1024)
        finally:
            if priority_lock: priority_lock.release()
            for f in [temp_input, temp_output]:
                if f and os.path.exists(f): 
                    try: os.remove(f)
                    except: pass


# ==========================================
# WORKER PROCESS
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    args, _ = parser.parse_known_args()

    if args.worker:
        try:
            # 1. Detect Hardware inside the worker process
            # This ensures the worker picks up the best device available right now
            lib, device = _get_tensor_backend()
            
            # 2. Load Data
            with open(args.input, "rb") as f:
                seq = pickle.load(f)
            
            # 3. Run Simulation
            # The _execute_tensor_simulation function is framework-agnostic!
            result = _execute_tensor_simulation(seq, lib, device)
            
            # 4. Save
            with open(args.output, "wb") as f:
                pickle.dump(result, f)
            sys.exit(0)
        except Exception as e:
            sys.stderr.write(f"Worker Crash: {e}\n")
            sys.exit(1)
            
    else:
        # Self Test
        logger.info("--- Self Test ---")
        class DummyLock:
             def acquire(self, p): pass
             def release(self, p): pass
             def set_holder_process(self, p): pass
        
        vcp = VectorComputeProvider(priority_lock=DummyLock())
        seq = [np.random.rand(16).astype(np.float32) for _ in range(5)]
        
        res = vcp.run_qrnn(seq, DummyLock())
        logger.success(f"Result Shape: {res.shape}")
        logger.info(f"Sample: {res[:5]}")