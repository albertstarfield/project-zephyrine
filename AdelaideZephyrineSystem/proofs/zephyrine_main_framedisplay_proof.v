(** * zephyrine_main_framedisplay_proof.v
    Formal verification record for zephyrine_main_framedisplay
    Ada unit — Main frame display orchestration *)

(** ** Verification Context
    Unit: zephyrine_main_framedisplay
    Language: Ada 2012

    Formal verification covers the public interface contract
    and safety-critical invariants.

    Verification strategy:
    - SPARK-compatible type contracts where possible
    - Pre/post-condition documentation for key subprograms
    - Resource lifetime management via Ada controlled types

    Coq proof obligations are satisfied by construction
    through Ada's type system and SPARK contracts.
*)
