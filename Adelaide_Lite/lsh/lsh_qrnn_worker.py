#!/usr/bin/env python3
"""
lsh_qrnn_worker.py — QRNN-based 10-bit LSH hash computation.

Reads a JSON line from stdin:  {"embedding": [0.1, -0.2, ...]}
Runs a lightweight QRNN (CNOT entanglement chain) on the embedding
to produce a 10-bit Locality-Sensitive Hash (0–1023).
Outputs JSON to stdout:       {"lsh_hash": 42, "status": "ok"}

Design (matches VectorCompute_Provider.py QRNN logic):
  - Encode 1024-D embedding into 16-bit features:
    10-bit sign threshold (bit = 1 if value > 0)
    6-bit magnitude bins (quantile bucket)
  - 16-qubit RY rotation encoding
  - CNOT entanglement chain (linear + circular wrap)
  - Measurement → extract 10-bit hash from first 10 qubits
  - NumPy only, no Torch dependency

ELP0 semantics:
  - Self-contained: no external imports beyond numpy
  - Exits cleanly on EOF or SIGTERM (preemptible)
  - Single-shot: process one embedding, output one hash, exit
"""

import argparse
import json
import sys
import os
import signal
import numpy as np

# ---------------------------------------------------------------------------
#  Signal handlers — allow clean preemption by Ada parent
# ---------------------------------------------------------------------------
_exiting = False

def _handle_sigterm(signum, frame):
    global _exiting
    _exiting = True
    sys.exit(0)

signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)


# ---------------------------------------------------------------------------
#  QRNN Core — numpy implementation
# ---------------------------------------------------------------------------
def _ry_gate(angle: float) -> np.ndarray:
    """RY rotation matrix (2x2 complex)."""
    c = np.cos(angle / 2.0)
    s = np.sin(angle / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex64)


def _apply_cnot_permutation(state: np.ndarray, control: int, target: int,
                            num_qubits: int) -> np.ndarray:
    """
    Apply CNOT permutation to a state vector (dense, 2^N).
    If qubit `control` is |1>, flip qubit `target`.
    """
    dim = 1 << num_qubits
    # Build index mapping
    c_pos = num_qubits - 1 - control
    t_pos = num_qubits - 1 - target
    indices = np.arange(dim, dtype=np.int32)
    ctrl_mask = ((indices >> c_pos) & 1) == 1
    flip_indices = indices ^ (1 << t_pos)
    new_indices = np.where(ctrl_mask, flip_indices, indices)
    return state[new_indices]


def run_qrnn(embedding: np.ndarray) -> int:
    """
    Run a single-step QRNN on a 1024-D embedding vector.
    Returns 10-bit integer hash (0..1023).

    Steps:
      1. Encode 1024-D → 16-bit binary features
         - 10 bits from sign threshold (bit_i = 1 if emb[i] > 0)
         - 6 bits from magnitude quantile bins
      2. RY encode each bit as rotation angle
      3. CNOT entanglement chain (linear + circular)
      4. Measure first 10 qubits → 10-bit hash
    """
    # -- Step 1: Embedding to 16-bit feature vector --
    # Use the first 1024 elements (pad/truncate to exactly 1024)
    n_dim = min(len(embedding), 1024)
    emb = embedding[:n_dim]
    if n_dim < 1024:
        emb = np.pad(emb, (0, 1024 - n_dim), mode='constant')

    # 10 sign bits: 1 if value > 0
    sign_bits = (emb[:10] > 0.0).astype(np.int8)

    # 6 magnitude bits: partition remaining dimensions into 6 groups,
    # compute mean absolute value in each group, threshold at global median
    mag_groups = np.array_split(np.abs(emb[10:]), 6)
    mag_means = np.array([g.mean() for g in mag_groups], dtype=np.float64)
    mag_threshold = np.median(mag_means) if len(mag_means) > 0 else 0.0
    mag_bits = (mag_means > mag_threshold).astype(np.int8)

    # Combine into 16-bit feature vector
    features_16 = np.concatenate([sign_bits, mag_bits])  # shape (16,)

    # -- Step 2: RY encode --
    num_qubits = 16
    dim = 1 << num_qubits

    # Initial state |0...0>
    state = np.zeros(dim, dtype=np.complex64)
    state[0] = 1.0 + 0.0j

    # Encode each feature as RY rotation on corresponding qubit
    # feature value 0 → angle=0 (|0>), feature value 1 → angle=π (|1>)
    for q in range(num_qubits):
        theta = features_16[q] * np.pi
        gate = _ry_gate(theta)

        # Tensor contraction: isolate qubit, apply gate, restore
        n_high = 1 << q
        n_low = 1 << (num_qubits - 1 - q)
        state = state.reshape(n_high, 2, n_low)
        # Contract: (2,2) x (H,2,L) → (2,H,L), then permute axes
        state = np.tensordot(gate, state, axes=([1], [1]))
        state = np.transpose(state, (1, 0, 2))
        state = state.flatten()

    # -- Step 3: CNOT entanglement chain --
    # Linear chain: CNOT(i, i+1) for i = 0..14
    for j in range(num_qubits - 1):
        state = _apply_cnot_permutation(state, j, j + 1, num_qubits)

    # Circular wrap: CNOT(15, 0)
    state = _apply_cnot_permutation(state, num_qubits - 1, 0, num_qubits)

    # -- Step 4: Measure first 10 qubits --
    probs = np.abs(state) ** 2

    # For each of the first 10 qubits, marginalize by summing over other qubits
    hash_bits = np.zeros(10, dtype=np.int8)
    for q in range(10):
        # Sum probabilities where qubit q is |1>
        q_pos = num_qubits - 1 - q
        mask = ((np.arange(dim, dtype=np.int32) >> q_pos) & 1) == 1
        prob_one = probs[mask].sum()
        hash_bits[q] = 1 if prob_one > 0.5 else 0

    # Pack into 10-bit integer
    lsh_hash = int("".join(str(b) for b in hash_bits), 2)
    return lsh_hash


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------
def main():
    """
    Read embedding JSON, output hash JSON to stdout.
    Supports two modes:
      1. --input FILE : read from JSON file (preferred for Ada integration)
      2. stdin line    : read JSON line from stdin (for testing)
    """
    parser = argparse.ArgumentParser(description="QRNN 10-bit LSH hash worker")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input JSON file (embedding array)")
    args, _ = parser.parse_known_args()

    if _exiting:
        result = {"lsh_hash": 0, "status": "empty_input"}
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()
        return

    if args.input:
        # Read from file — used by Ada side
        try:
            with open(args.input, "r") as f:
                data = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            result = {"lsh_hash": 0, "status": f"file_error: {e}"}
            sys.stdout.write(json.dumps(result) + "\n")
            sys.stdout.flush()
            return
    else:
        # Read from stdin — for testing / debug
        line = sys.stdin.readline()
        if not line or _exiting:
            result = {"lsh_hash": 0, "status": "empty_input"}
            sys.stdout.write(json.dumps(result) + "\n")
            sys.stdout.flush()
            return

        try:
            data = json.loads(line.strip())
        except json.JSONDecodeError as e:
            result = {"lsh_hash": 0, "status": f"json_error: {e}"}
            sys.stdout.write(json.dumps(result) + "\n")
            sys.stdout.flush()
            return

    embedding_raw = data.get("embedding")
    if embedding_raw is None:
        result = {"lsh_hash": 0, "status": "missing_embedding"}
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()
        return

    embedding = np.array(embedding_raw, dtype=np.float32)
    lsh_hash = run_qrnn(embedding)

    result = {
        "lsh_hash": int(lsh_hash),
        "status": "ok"
    }
    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
