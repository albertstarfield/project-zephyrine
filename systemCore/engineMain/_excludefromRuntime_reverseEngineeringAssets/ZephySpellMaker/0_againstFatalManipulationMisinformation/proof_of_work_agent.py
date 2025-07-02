#! /usr/bin/env python3

import argparse
import base64
import json
import hashlib
import time
from ecdsa import SECP256k1, SigningKey, VerifyingKey
from kyber import Kyber1024

# ---
# Proof-of-Work PQC Decryption Agent
#
# This script implements a cryptographic puzzle where the solution to a
# classic hard problem (ECDLP) is used to seed a post-quantum (PQC) key.
#
# ARCHITECTURE:
# 1. SETUP: A "target" Elliptic Curve public key (P) is created from a
#    secret number (k). This (P) is the puzzle. The solution (k) is used
#    to generate a Kyber PQC public/private key pair. The PQC public key
#    is saved, but the PQC private key is discarded.
#
# 2. ENCRYPT: A payload is encrypted using the public PQC key.
#
# 3. RUN AGENT: The agent is given the target public key (P). It begins
#    a brute-force search for the secret number `k` by checking every
#    possible value (1, 2, 3, ...). This is the CPU-intensive "work".
#
# 4. UNLOCK: When the agent finds the correct `k` such that `k * G = P`,
#    it uses that `k` to re-generate the PQC private key and decrypt the
#    payload.
# ---

def setup_puzzle(solution_k: int, config_path="puzzle_config.json"):
    """
    Creates the cryptographic puzzle for the agent to solve.
    - solution_k: The secret integer the agent must find. This determines the difficulty.
    """
    print(f"🛠️  Setting up puzzle with solution k = {solution_k}")
    if solution_k <= 0:
        raise ValueError("Solution k must be a positive integer.")

    # --- Part 1: The Elliptic Curve Puzzle (The Lock) ---
    curve = SECP256k1
    # Create the private key `k` and corresponding public key `P`
    # P = k * G
    ecdsa_private_key = SigningKey.from_secret_exponent(solution_k, curve=curve)
    ecdsa_public_key = ecdsa_private_key.get_verifying_key()
    target_public_key_p_bytes = ecdsa_public_key.to_string("uncompressed")
    
    print(f"    - Target EC Public Key (P) generated.")
    
    # --- Part 2: The PQC Key Generation (The Key inside the Lock) ---
    # The solution `k` will be the master seed for the PQC key.
    # We hash it to ensure it's a good, uniformly distributed seed.
    k_bytes = solution_k.to_bytes(32, 'big', signed=False)
    pqc_seed = hashlib.sha256(k_bytes).digest()

    # Generate the Kyber key pair from this seed.
    pqc_public_key, _ = Kyber1024.keygen(pqc_seed)
    print(f"    - PQC Key Pair derived from solution k.")

    # --- Part 3: Save the public-facing puzzle configuration ---
    # The agent will need the target EC public key and the PQC public key.
    # The solution `k` and the PQC private key are NOT saved.
    config_data = {
        "description": "Proof-of-Work Puzzle Config",
        "ecdsa_curve": "SECP256k1",
        "target_ec_public_key_hex": target_public_key_p_bytes.hex(),
        "pqc_public_key_b64": base64.b64encode(pqc_public_key).decode('utf-8')
    }
    
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=4)
    
    print(f"✅ Puzzle configuration saved to '{config_path}'")
    print("    - The agent must now find the original `k` to unlock the payload.")

def encrypt_payload(config_path, plaintext, output_path):
    """Encrypts a payload using the PQC public key from the puzzle config."""
    print("🔒 Encrypting payload...")
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    pqc_public_key = base64.b64decode(config_data["pqc_public_key_b64"])
    
    ciphertext, _ = Kyber1024.enc(pqc_public_key, plaintext.encode('utf-8'))
    b64_ciphertext = base64.b64encode(ciphertext).decode('utf-8')
    
    with open(output_path, "w") as f:
        f.write(b64_ciphertext)
    
    print(f"✅ Payload encrypted and saved to '{output_path}'")


def run_agent(config_path, payload_path):
    """
    The main agent logic. It loads the puzzle and starts the brute-force search.
    This function will use 100% of one CPU core.
    """
    print("🚀 Agent activated. Loading puzzle...")
    with open(config_path, "r") as f:
        config_data = json.load(f)
    with open(payload_path, "r") as f:
        encrypted_payload = f.read()
        
    target_ec_public_key_bytes = bytes.fromhex(config_data["target_ec_public_key_hex"])
    target_ec_public_key = VerifyingKey.from_string(target_ec_public_key_bytes, curve=SECP256k1)
    
    curve = SECP256k1
    generator = curve.generator
    
    print(f"🔥 Starting brute-force search for the secret 'k'. This will use 100% CPU.")
    print(f"   Searching for k such that k * G == {target_ec_public_key_bytes.hex()[:20]}...")
    
    start_time = time.time()
    # The brute-force loop. It will run from k=1 up to the order of the curve.
    for k_guess in range(1, curve.order):
        # This is the core computational work: Elliptic Curve point multiplication
        p_guess = k_guess * generator
        
        # Print progress to show it's working
        if k_guess % 100000 == 0:
            elapsed = time.time() - start_time
            rate = k_guess / elapsed
            print(f"   ...checked {k_guess:,} keys. ({rate:,.0f} keys/sec)")
            
        if p_guess == target_ec_public_key.pubkey.point:
            elapsed = time.time() - start_time
            print(f"\n🎉 SUCCESS! Solution found after {elapsed:.2f} seconds.")
            print(f"    - Found secret k = {k_guess}")
            
            # --- UNLOCKING THE PQC BOX ---
            print("    - Re-deriving PQC private key from solution...")
            k_bytes = k_guess.to_bytes(32, 'big', signed=False)
            pqc_seed = hashlib.sha256(k_bytes).digest()
            _, pqc_private_key = Kyber1024.keygen(pqc_seed)
            
            print("    - Decrypting payload with re-derived key...")
            ciphertext = base64.b64decode(encrypted_payload)
            decrypted_bytes = Kyber1024.dec(pqc_private_key, ciphertext)
            
            print("\n--- DECRYPTED PAYLOAD ---")
            print(decrypted_bytes.decode('utf-8'))
            print("---       END       ---\n")
            return

    print("❌ Search completed without finding a solution. This should not happen if setup was correct.")


def main():
    parser = argparse.ArgumentParser(description="Proof-of-Work PQC Agent")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # --- Setup Command ---
    parser_setup = subparsers.add_parser("setup", help="Create the puzzle for the agent.")
    parser_setup.add_argument("--solution", required=True, type=int, help="The secret integer 'k' to base the puzzle on. Higher is harder.")
    parser_setup.add_argument("--config-out", default="puzzle_config.json")
    
    # --- Encrypt Command ---
    parser_encrypt = subparsers.add_parser("encrypt", help="Encrypt a payload with the puzzle's PQC key.")
    parser_encrypt.add_argument("--config-in", default="puzzle_config.json")
    parser_encrypt.add_argument("--payload", required=True, type=str)
    parser_encrypt.add_argument("--output", default="payload.enc")
    
    # --- Run Agent Command ---
    parser_run = subparsers.add_parser("run-agent", help="Activate the agent to solve the puzzle.")
    parser_run.add_argument("--config-in", default="puzzle_config.json")
    parser_run.add_argument("--payload-in", default="payload.enc")

    args = parser.parse_args()
    
    if args.command == "setup":
        setup_puzzle(args.solution, args.config_out)
    elif args.command == "encrypt":
        encrypt_payload(args.config_in, args.payload, args.output)
    elif args.command == "run-agent":
        run_agent(args.config_in, args.payload_in)

if __name__ == "__main__":
    main()