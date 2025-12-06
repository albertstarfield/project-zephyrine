#! /usr/bin/env python3

import argparse
import base64
import json
import hashlib
import time
import os
import random
from datetime import datetime
from ecdsa import SECP256k1, SigningKey
from kyber import Kyber1024                  ### MODIFIED ###
from reedsolo import RSCodec

# ---
# The Hydra Agent (Portable Version)
#
# This version uses the pure-python 'kyber-py' library for PQC,
# removing the need for a C compiler and making it fully portable.
# ---

# --- CONFIGURATION (Identical) ---
TIME_WINDOW_START = (3, 19, 0)
TIME_WINDOW_END = (4, 12, 0)
N_FRAGMENTS = 10
K_REQUIRED = 6
BEGIN_MARKER = b"--BEGIN HYDRA FRAGMENT--"
END_MARKER = b"--END HYDRA FRAGMENT--"
CURVE = SECP256k1
RSC = RSCodec(N_FRAGMENTS - K_REQUIRED)

# --- HELPER FUNCTIONS FOR NEW PQC LOGIC ---

def xor_bytes(a, b):
    """Simple XOR function for one-time pad encryption."""
    return bytes(x ^ y for x, y in zip(a, b))

def pqc_encrypt(public_key, plaintext_bytes):
    """Encrypts plaintext using Kyber KEM and a one-time pad."""
    # 1. Encapsulate a shared secret. `c` is the ciphertext to send.
    c, shared_secret = Kyber1024.enc(public_key)
    # 2. "Encrypt" the payload using the shared secret as a one-time pad.
    encrypted_payload = xor_bytes(plaintext_bytes, shared_secret)
    # 3. Return the Kyber ciphertext and the XORed payload together.
    return c, encrypted_payload

def pqc_decrypt(private_key, c, encrypted_payload):
    """Decrypts using Kyber KEM and a one-time pad."""
    # 1. Decapsulate to get the same shared secret.
    shared_secret = Kyber1024.dec(c, private_key)
    # 2. "Decrypt" the payload by XORing again with the secret.
    plaintext_bytes = xor_bytes(encrypted_payload, shared_secret)
    return plaintext_bytes


# --- CORE LOGIC (Mostly Unchanged, just uses new pqc functions) ---

def is_window_open():
    now = datetime.now()
    day, hour, min = now.weekday(), now.hour, now.minute
    start_day, start_hour, start_min = TIME_WINDOW_START
    end_day, end_hour, end_min = TIME_WINDOW_END
    if start_day == end_day: return start_day == day and (start_hour, start_min) <= (hour, min) <= (end_hour, end_min)
    else:
        if day == start_day and (hour, min) >= (start_hour, start_min): return True
        if day == end_day and (hour, min) <= (end_hour, end_min): return True
        if start_day < day < end_day: return True
    return False

def scavenge_fragments():
    print("🔎 Scavenging for Hydra fragments...")
    fragments, locations = [], {}
    for root, _, files in os.walk('.'):
        for filename in files:
            if filename.endswith((".py", ".json")): continue
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, 'rb') as f: content = f.read()
                start_idx = content.find(BEGIN_MARKER)
                if start_idx != -1:
                    end_idx = content.find(END_MARKER, start_idx)
                    if end_idx != -1:
                        fragments.append(content[start_idx + len(BEGIN_MARKER):end_idx])
                        locations[filepath] = (start_idx, end_idx + len(END_MARKER))
            except (IOError, PermissionError): continue
    print(f"   Found {len(fragments)} fragments across {len(locations)} files.")
    return fragments, locations

def reconstruct_state(fragments):
    if len(fragments) < K_REQUIRED: return None
    print(f"🧩 Reconstructing state from {len(fragments)} fragments (need {K_REQUIRED})...")
    try:
        reconstructed_data = RSC.decode([bytearray(f) for f in fragments])[0]
        return json.loads(reconstructed_data.decode('utf-8'))
    except Exception as e:
        print(f"   ❌ Reconstruction failed: {e}")
        return None

def disperse_and_hide_state(state_json_str):
    print(f"🐍 Dispersing new state into {N_FRAGMENTS} fragments...")
    host_files = []
    for root, _, files in os.walk('.'):
        for filename in files:
            if filename.endswith(('.txt', '.log', '.md')):
                host_files.append(os.path.join(root, filename))
    if not host_files:
        print("   ❌ No suitable host files found!")
        return
    fragments = RSC.encode(state_json_str.encode('utf-8'))
    for i, fragment in enumerate(fragments):
        wrapped_fragment = BEGIN_MARKER + fragment + END_MARKER
        host_file = random.choice(host_files)
        print(f"   - Hiding fragment {i+1}/{N_FRAGMENTS} in '{host_file}'")
        try:
            with open(host_file, 'ab') as f: f.write(wrapped_fragment)
        except (IOError, PermissionError) as e:
            print(f"     - Failed to write to {host_file}: {e}")

def cleanup_old_fragments(locations):
    print("🧹 Cleaning up old fragments...")
    for filepath, (start, end) in locations.items():
        try:
            with open(filepath, 'rb') as f: content = f.read()
            with open(filepath, 'wb') as f:
                f.write(content[:start])
                f.write(content[end:])
            print(f"   - Removed old fragment from '{filepath}'")
        except (IOError, PermissionError) as e:
            print(f"   - Failed to clean {filepath}: {e}")

def solve_puzzle_and_ratchet(state_data):
    puzzle_index = state_data.get('puzzle_index', 'N/A')
    target_point = SigningKey.from_string(bytes.fromhex(state_data['target_ec_public_key_hex']), curve=CURVE).verifying_key.pubkey.point

    print(f"🔥 Starting brute-force for Puzzle {puzzle_index}. This will use 100% CPU.")
    start_time = time.time()
    solution_k = -1
    for k_guess in range(1, CURVE.order):
        if k_guess % 1000000 == 0: print(f"   ...checked {k_guess:,} keys...")
        if k_guess * CURVE.generator == target_point:
            solution_k = k_guess
            break
    if solution_k == -1: return None, None, "Search failed."
    elapsed = time.time() - start_time
    print(f"\n🎉 SUCCESS! Solved Puzzle {puzzle_index} in {elapsed:.2f} seconds.")
    print(f"    - Found solution k_{puzzle_index} = {solution_k}")

    # Decrypt payload using the new PQC logic
    k_bytes = solution_k.to_bytes(32, 'big')
    pqc_seed = hashlib.sha256(k_bytes).digest()
    
    # We don't need the public key here, just the private one
    _ , pqc_priv_key = Kyber1024.keypair(pqc_seed) ### MODIFIED ###
    
    # The stored payload now has two parts
    kyber_ciphertext_c = base64.b64decode(state_data['kyber_ciphertext_c_b64'])
    encrypted_payload = base64.b64decode(state_data['encrypted_payload_b64'])
    decrypted_payload = pqc_decrypt(pqc_priv_key, kyber_ciphertext_c, encrypted_payload).decode('utf-8')
    
    # Ratchet forward
    next_k_seed = hashlib.sha256(k_bytes).digest()
    next_k = int.from_bytes(next_k_seed, 'big') % CURVE.order
    if next_k == 0: next_k = 1

    # Create new state using the new PQC logic
    ec_priv_key = SigningKey.from_secret_exponent(next_k, curve=CURVE)
    pqc_seed_next = hashlib.sha256(next_k.to_bytes(32, 'big')).digest()
    pqc_pub_key_next, _ = Kyber1024.keypair(pqc_seed_next) ### MODIFIED ###
    c_next, encrypted_payload_next = pqc_encrypt(pqc_pub_key_next, decrypted_payload.encode('utf-8'))
    
    new_state = {
        "puzzle_index": puzzle_index + 1,
        "target_ec_public_key_hex": ec_priv_key.get_verifying_key().to_string("uncompressed").hex(),
        "kyber_ciphertext_c_b64": base64.b64encode(c_next).decode('utf-8'), ### MODIFIED ###
        "encrypted_payload_b64": base64.b64encode(encrypted_payload_next).decode('utf-8')
    }
    
    return decrypted_payload, new_state, f"New solution will be k_{puzzle_index + 1} = {next_k}"

def main():
    parser = argparse.ArgumentParser(description="The Hydra Agent (Portable)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    parser_setup = subparsers.add_parser("setup", help="Create the initial dispersed state.")
    parser_setup.add_argument("--k0", required=True, type=int)
    parser_setup.add_argument("--payload", required=True, type=str)

    subparsers.add_parser("run-agent", help="Activate one cycle of the Hydra agent.")
    
    parser_hosts = subparsers.add_parser("create-hosts", help="Create dummy files for hiding data.")
    parser_hosts.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    
    if args.command == "create-hosts":
        print(f"Creating {args.count} dummy host files...")
        for i in range(args.count):
            with open(f"dummy_file_{i}.txt", "w") as f: f.write(f"This is a dummy file.\n" * (10 + i))
        print("Done.")
        return

    if args.command == "setup":
        ec_priv_key = SigningKey.from_secret_exponent(args.k0, curve=CURVE)
        pqc_seed = hashlib.sha256(args.k0.to_bytes(32, 'big')).digest()
        pqc_pub_key, _ = Kyber1024.keypair(pqc_seed) ### MODIFIED ###
        
        c, encrypted_payload = pqc_encrypt(pqc_pub_key, args.payload.encode('utf-8'))
        
        initial_state = {
            "puzzle_index": 0,
            "target_ec_public_key_hex": ec_priv_key.get_verifying_key().to_string("uncompressed").hex(),
            "kyber_ciphertext_c_b64": base64.b64encode(c).decode('utf-8'), ### MODIFIED ###
            "encrypted_payload_b64": base64.b64encode(encrypted_payload).decode('utf-8')
        }
        disperse_and_hide_state(json.dumps(initial_state))
        print("\n✅ Initial Hydra state dispersed successfully.")
        return

    if args.command == "run-agent":
        print("--- HYDRA AGENT ACTIVATION SEQUENCE ---")
        if not is_window_open():
            print(f"⏳ Time-lock engaged. Standing by.")
            return
        
        print("✅ Time-lock disengaged. Window is open.")
        fragments, locations = scavenge_fragments()
        
        if len(fragments) < K_REQUIRED:
            print(f"❌ Insufficient data found ({len(fragments)}/{K_REQUIRED}). Aborting.")
            return
            
        state_data = reconstruct_state(fragments)
        if not state_data:
            print("❌ State reconstruction failed. Aborting.")
            return
            
        decrypted_payload, new_state, ratchet_msg = solve_puzzle_and_ratchet(state_data)
        
        if not decrypted_payload:
            print(f"❌ Puzzle solving failed: {new_state}. Aborting.")
            return

        print("\n--- DECRYPTED PAYLOAD ---")
        print(decrypted_payload)
        print("-------------------------\n")
        
        print(f"🔒 Ratcheting forward... {ratchet_msg}")
        
        disperse_and_hide_state(json.dumps(new_state))
        cleanup_old_fragments(locations)

        print("\n--- HYDRA CYCLE COMPLETE. AGENT IS DORMANT. ---")

if __name__ == "__main__":
    main()