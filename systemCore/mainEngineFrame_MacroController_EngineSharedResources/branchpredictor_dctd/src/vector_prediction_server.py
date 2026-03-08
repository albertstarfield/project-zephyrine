#!/usr/bin/env python3
import zmq
import json
import random
import traceback
import sys

# Global Backend Cache
BACKEND = None
HAS_QRACK = False

def initialize_backend():
    """
    Initialize the quantum backend once at startup.
    """
    global BACKEND, HAS_QRACK
    print("Initializing Vector Prediction Backend...")
    
    try:
        # Try direct Qrack import
        try:
            from qiskit.providers.qrack import QasmSimulator
            BACKEND = QasmSimulator()
            HAS_QRACK = True
            print("Backend Loaded: Qrack Quantum Simulator")
            return True
        except ImportError:
            print("Qrack not available, trying provider pattern...")

        # Try provider pattern
        from qiskit.providers.qrack import Qrack
        BACKEND = Qrack.get_backend('qasm_simulator')
        HAS_QRACK = True
        print("Backend Loaded: Qrack via Provider")
        return True
        
    except Exception as e:
        print(f"Error loading Qrack: {e}")
        print("Falling back to Mock Mode")
        HAS_QRACK = False
        return False

def run_prediction(input_bytes):
    """
    Main prediction logic.
    """
    global BACKEND, HAS_QRACK
    
    try:
        # Parse input
        payload = json.loads(input_bytes.decode('utf-8'))
        vectors = payload.get("vectors", [])
        input_seed = len(vectors) if vectors else 0
            
    except Exception as e:
        print(f"Input parsing error: {e}")
        input_seed = 0

    predicted_hash = input_seed
    
    if HAS_QRACK and BACKEND:
        try:
            from qiskit import QuantumCircuit
            
            # Create quantum circuit
            qc = QuantumCircuit(3)
            
            # Encoding based on input seed
            if input_seed % 2 == 1:
                qc.x(0)
            if (input_seed // 2) % 2 == 1:
                qc.x(1)
                
            # Quantum operations
            qc.h(0)
            qc.cx(0, 1)
            qc.cx(1, 2)
            qc.rx(0.1, 0)
            qc.measure_all()
            
            # Execute
            job = BACKEND.run(qc, shots=1)
            result = job.result()
            counts = result.get_counts()
            
            # Get quantum outcome
            outcome_str = list(counts.keys())[0]
            outcome_int = int(outcome_str, 2)
            predicted_hash += outcome_int
            
        except Exception as e:
            print(f"Quantum simulation failed: {e}")
            predicted_hash += 1
    else:
        # Mock fallback
        predicted_hash += random.randint(0, 7)

    # Prepare response
    response = {
        "predicted_lsh_hash": predicted_hash,
        "predicted_time_horizon": "+5 mins",
        "debug_input_count": input_seed,
        "quantum_backend": "qrack" if HAS_QRACK else "mock"
    }
    
    return json.dumps(response).encode('utf-8')

def main():
    """ZMQ REP server main loop"""
    # Initialize backend
    if not initialize_backend():
        print("Warning: Running in mock mode without quantum backend")
    
    # Setup ZMQ
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    
    # Use the same socket path as before
    socket_path = "ipc://celestial_timestream_vector_helper.socket"
    
    try:
        socket.bind(socket_path)
        print(f"ZMQ Server listening on {socket_path}")
        print("Ready for requests...")
        
        while True:
            # Wait for request
            message = socket.recv()
            print(f"Received request ({len(message)} bytes)")
            
            # Process prediction
            response = run_prediction(message)
            
            # Send response
            socket.send(response)
            print(f"Sent response ({len(response)} bytes)")
            
    except KeyboardInterrupt:
        print("Server shutting down...")
    except Exception as e:
        print(f"Server error: {e}")
        traceback.print_exc()
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    main()