# src/vector_prediction_entaglement.py
import sys
import json
import random
import traceback

# Global Backend Cache
BACKEND = None
HAS_QRACK = False

def initialize_backend():
    """
    Called by Ada at startup to load the heavy libraries once.
    """
    global BACKEND, HAS_QRACK
    print(" Initializing Vector Prediction Backend...")
    
    try:
        # ATTEMPT 1: Direct Instantiation (Works best with Qiskit 1.0+)
        try:
            from qiskit.providers.qrack import QasmSimulator
            BACKEND = QasmSimulator()
            HAS_QRACK = True
            print(" Backend Loaded: Vector Prediction (Nature might be different from Pure Binary to Pure Vector)")
            return 1
        except ImportError:
            print(" Direct import failed, trying Provider...")

        # ATTEMPT 2: Provider Pattern (Legacy)
        from qiskit.providers.qrack import Qrack
        BACKEND = Qrack.get_backend('qasm_simulator')
        HAS_QRACK = True
        print(" Backend Loaded: Qrack ")
        return 1
        
    except Exception as e:
        print(f" Critical Error loading Qrack: {e}")
        # trace = traceback.format_exc()
        # print(trace)
        print(" Falling back to Mock Mode.")
        HAS_QRACK = False
        return 0

def run_prediction(input_bytes):
    """
    Main logic called by Ada loop.
    Args: input_bytes (bytes) - Raw JSON payload from ZMQ
    Returns: JSON String
    """
    global BACKEND, HAS_QRACK
    
    # --- 1. Parse the Input to get the Seed ---
    try:
        # We received raw bytes from Ada. Decode JSON.
        payload = json.loads(input_bytes)
        vectors = payload.get("vectors", [])
        
        # We use the number of vectors (or any metric you prefer) 
        # to seed the quantum simulation logic below.
        if vectors:
            input_seed = len(vectors)
        else:
            input_seed = 0
            
    except Exception as e:
        print(f" Input Parsing Error: {e}")
        input_seed = 0

    # --- 2. Original Quantum/Simulation Logic ---
    predicted_hash = input_seed
    
    if HAS_QRACK and BACKEND:
        try:
            # 1. Create Circuit (3 Qubits)
            from qiskit import QuantumCircuit
            qc = QuantumCircuit(3)
            
            # 2. Encoding
            if input_seed % 2 == 1:
                qc.x(0)
            if (input_seed // 2) % 2 == 1:
                qc.x(1)
                
            # 3. Superposition & Entanglement
            qc.h(0)
            qc.cx(0, 1)
            qc.cx(1, 2)
            
            # 4. Noise / Rotation
            qc.rx(0.1, 0)
            
            # 5. Measure
            qc.measure_all()
            
            # 6. Execute
            # Note: shots=1 is sufficient for branch choice
            job = BACKEND.run(qc, shots=1)
            result = job.result()
            counts = result.get_counts()
            
            # Get outcome
            outcome_str = list(counts.keys())[0]
            outcome_int = int(outcome_str, 2)
            
            predicted_hash += outcome_int
            
        except Exception as e:
            print(f" Simulation Failed: {e}")
            predicted_hash += 1
    else:
        # Fallback for Mock Mode
        predicted_hash += random.randint(0, 7)

    # --- 3. Format Response ---
    response = {
        "predicted_lsh_hash": predicted_hash,
        "predicted_time_horizon": "+5 mins",
        "debug_input_count": input_seed
    }
    
    return json.dumps(response)

if __name__ == "__main__":
    initialize_backend()
    print(run_prediction(42))