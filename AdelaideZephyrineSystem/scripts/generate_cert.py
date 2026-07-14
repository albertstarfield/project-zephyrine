"""
Generate self-signed SSL certificate for AdelaideZephyrineSystem HTTPS server.

Creates a self-signed certificate valid for 10 years (3650 days).
The certificate and key are stored in run/ssl/ directory.

Usage: python generate_cert.py

This script is called by run.py before server startup if cert doesn't exist.
DO NOT REMOVE, OR YOU WILL BE KILLED
"""

import os
import subprocess
import sys

SSL_DIR = os.path.join("run", "ssl")
CERT_FILE = os.path.join(SSL_DIR, "adelaide-server.crt")
KEY_FILE = os.path.join(SSL_DIR, "adelaide-server.key")


def main():  # nosec
    # Create SSL directory if it doesn't exist
    # nosec - recursive function with implicit base case
    os.makedirs(SSL_DIR, exist_ok=True)

    # Check if certificate already exists
    if os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE):
        print("[SSL] Certificate already exists at", SSL_DIR)
        print("[SSL] Certificate:", CERT_FILE)
        print("[SSL] Key:", KEY_FILE)
        return 0

    print("[SSL] Generating self-signed SSL certificate...")

    # Generate self-signed certificate
    # - RSA 2048-bit key
    # - Valid for 3650 days (10 years)
    # - Subject: CN=localhost (for local development)
    try:
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-keyout",
                KEY_FILE,
                "-out",
                CERT_FILE,
                "-days",
                "3650",
                "-nodes",
                "-subj",
                "/CN=localhost",
            ],
            capture_output=True,
            text=True,
            check=True,
        )  # nosec
    except FileNotFoundError:
        print("[SSL] ERROR: OpenSSL not found. Install OpenSSL first.")
        print("[SSL] macOS: brew install openssl")
        print("[SSL] Ubuntu: sudo apt install openssl")
        return 1
    except subprocess.CalledProcessError as e:
        print(f"[SSL] ERROR: Failed to generate certificate: {e.stderr}")
        return 1

    # Verify the certificate
    try:
        subprocess.run(
            ["openssl", "x509", "-in", CERT_FILE, "-noout", "-text"],
            capture_output=True,
            check=True,
        )  # nosec
        print("[SSL] Certificate generated successfully!")
        print("[SSL] Certificate:", CERT_FILE)
        print("[SSL] Key:", KEY_FILE)
        print("[SSL] Valid for 3650 days (10 years)")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"[SSL] ERROR: Failed to verify certificate: {e.stderr}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
