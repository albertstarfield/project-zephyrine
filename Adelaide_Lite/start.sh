#!/bin/bash
# start.sh - Verified Startup Script for Adelaide Server

echo "[*] Verifying codebase with GNATprove (Level 4)..."
alr exec -- gnatprove -P adelaide_lite.gpr --level=4 --prover=cvc5,z3,altergo --timeout=60 --memlimit=2000 --steps=0 --counterexamples=on --report=fail

if [ $? -eq 0 ]; then
    echo "[ok] Codebase verified securely! Starting Adelaide Server..."
    alr exec -- ./bin/adelaide_server "$@"
else
    echo "[!] CRITICAL: SPARK Proofing Failed! Refusing to start the server until formal verification passes."
    exit 1
fi
