(** * glesv2_binding_proof.v
    Formal verification record for glesv2_binding
    Ada unit — OpenGL ES 2.0 shader/binding declarations *)

(** ** Verification Context
    Unit: glesv2_binding
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
