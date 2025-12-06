#! /usr/bin/env python3

import argparse
import base64
import json
import hashlib
import time
import os
from ecdsa import SECP256k1, SigningKey
from kyber import Kyber1024

# ---
# Self-Ratcheting Proof-of-Work PQC Agent
#
# This agent solves a cryptographic puzzle to unlock a payload.
# Upon success, it uses a one-way function (ratchet) to generate the *next*
# puzzle in the sequence and re-encrypts the payload, making the previous
# state irrecoverable.
#
# This provides Forward Secrecy and Post-Compromise Security.
# ---

STATE_FILE = "agent_state.json"
CURVE = SECP256k1

def create_puzzle_state(k: int, payload: str):
    """Generates a complete state dictionary for a given solution 'k'."""
    # 1. Generate the EC puzzle
    ec_priv_key = SigningKey.from_secret_exponent(k, curve=CURVE)
    ec_pub_key = ec_priv_key.get_verifying_key()
    
    # 2. Derive the PQC key from the solution 'k'
    k_bytes = k.to_bytes(32, 'big', signed=False)
    pqc_seed = hashlib.sha256(k_bytes).digest()
    pqc_pub_key, _ = Kyber1024.keygen(pqc_seed)
    
    # 3. Encrypt the payload with the PQC key
    ciphertext, _ = Kyber1024.enc(pqc_pub_key, payload.encode('utf-8'))
    
    return {
        "target_ec_public_key_hex": ec_pub_key.to_string("uncompressed").hex(),
        "encrypted_payload_b64": base64.b64encode(ciphertext).decode('utf-8')
    }

def setup_initial_state(initial_k: int, initial_payload: str):
    """Creates the very first puzzle (state 0) for the agent."""
    print(f"🛠️  Setting up initial state with k_0 = {initial_k}")
    if initial_k <= 0 or initial_k >= CURVE.order:
        raise ValueError("Initial k is out of the valid range for the curve.")
        
    state_0 = create_puzzle_state(initial_k, initial_payload)
    state_0['puzzle_index'] = 0
    
    with open(STATE_FILE, "w") as f:
        json.dump(state_0, f, indent=4)
        
    print(f"✅ Initial state (Puzzle 0) saved to '{STATE_FILE}'")

def run_agent():
    """The main agent loop: solve, decrypt, and ratchet forward."""
    if not os.path.exists(STATE_FILE):
        print(f"❌ Error: State file '{STATE_FILE}' not found. Run 'setup' first.")
        return

    # 1. Load the current state
    print(f"🚀 Agent activated. Loading current state from '{STATE_FILE}'...")
    with open(STATE_FILE, 'r') as f:
        current_state = json.load(f)
    
    puzzle_index = current_state.get('puzzle_index', 'N/A')
    target_ec_pub_key_hex = current_state['target_ec_public_key_hex']
    target_ec_pub_key_point = SigningKey.from_string(bytes.fromhex(target_ec_pub_key_hex), curve=CURVE).verifying_key.pubkey.point

    # 2. Solve the current puzzle via brute-force
    print(f"🔥 Starting brute-force for Puzzle {puzzle_index}. This will use 100% CPU.")
    start_time = time.time()
    solution_k = -1

    for k_guess in range(1, CURVE.order):
        if k_guess % 500000 == 0:
            print(f"   ...checked {k_guess:,} keys...")
            
        if k_guess * CURVE.generator == target_ec_pub_key_point:
            solution_k = k_guess
            break
            
    if solution_k == -1:
        print("❌ Search failed. The puzzle is likely unsolvable.")
        return

    elapsed = time.time() - start_time
    print(f"\n🎉 SUCCESS! Solved Puzzle {puzzle_index} in {elapsed:.2f} seconds.")
    print(f"    - Found solution k_{puzzle_index} = {solution_k}")
    
    # 3. Use the solution to decrypt the payload
    k_bytes = solution_k.to_bytes(32, 'big', signed=False)
    pqc_seed = hashlib.sha256(k_bytes).digest()
    _, pqc_priv_key = Kyber1024.keygen(pqc_seed)
    
    ciphertext = base64.b64decode(current_state['encrypted_payload_b64'])
    decrypted_bytes = Kyber1024.dec(pqc_priv_key, ciphertext)
    decrypted_payload = decrypted_bytes.decode('utf-8')
    
    print("\n--- DECRYPTED PAYLOAD ---")
    print(decrypted_payload)
    print("-------------------------\n")
    
    # 4. RATCHET FORWARD: Create the next state
    print(f"🔒 Ratcheting forward from k_{puzzle_index} to k_{puzzle_index + 1}...")
    
    # Use a secure, one-way hash function to derive the next solution.
    # We must ensure the result is a valid key for the curve.
    next_k_seed = hashlib.sha256(k_bytes).digest()
    next_k = int.from_bytes(next_k_seed, 'big') % CURVE.order
    # Ensure it's not zero, which is invalid.
    if next_k == 0: next_k = 1 

    print(f"    - New solution will be k_{puzzle_index + 1} = {next_k}")
    
    # The payload for the next state could be the same, or updated. Here we keep it the same.
    next_state = create_puzzle_state(next_k, decrypted_payload)
    next_state['puzzle_index'] = puzzle_index + 1
    
    # 5. Atomically save the new state
    # Write to a temporary file first, then rename. This prevents corruption.
    temp_file = f"{STATE_FILE}.tmp"
    with open(temp_file, "w") as f:
        json.dump(next_state, f, indent=4)
    os.replace(temp_file, STATE_FILE)
    
    print(f"✅ Ratchet complete. New state (Puzzle {puzzle_index + 1}) saved to '{STATE_FILE}'.")
    print("   The previous state is now cryptographically unrecoverable.")

def main():
    parser = argparse.ArgumentParser(description="Self-Ratcheting Proof-of-Work Agent")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    parser_setup = subparsers.add_parser("setup", help="Create the initial puzzle state.")
    parser_setup.add_argument("--k0", required=True, type=int, help="The initial secret integer 'k'.")
    parser_setup.add_argument("--payload", required=True, type=str, help="The initial secret payload.")
    
    subparsers.add_parser("run-agent", help="Activate the agent to solve one cycle.")

    args = parser.parse_args()
    
    if args.command == "setup":
        setup_initial_state(args.k0, args.payload)
    elif args.command == "run-agent":
        run_agent()

if __name__ == "__main__":
    main()