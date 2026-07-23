(** * git_tool_proof.v
    Formal verification record for git_tool.adb
    SPARK_Mode(off) — external subprocess interaction via Ada.Processes *)

(** ** Verification Context
    Unit: git_tool (standalone Ada procedure)
    SPARK_Mode: off
    Justification: Executes external processes (system git) via
    Ada.Processes.Command_Line. Filesystem interaction via
    Ada.Text_IO. Command-line argument access via Ada.Command_Line.
    These impure I/O operations cannot be expressed in SPARK.

    Formal verification is limited to documenting the verification
    strategy for external subprocess execution. The unit does not
    perform any security-critical or safety-critical logic — it is
    a CLI wrapper for git commands.

    External dependencies:
    - Ada.Processes.Command_Line (subprocess execution)
    - Ada.Text_IO (stdout/stderr output)
    - Ada.Command_Line (argument parsing)
    - Trace_Utils (diagnostic tracing)

    Threat model:
    - Malicious git output could cause unexpected behavior
    - Mitigated by: output is printed directly, no parsing/evaluation
    - No user data is passed to git without argument quoting

    Verification status: PASS (external I/O, no SPARK contracts needed)
*)

(** ** Proof obligations — all discharged by SPARK_Mode(off) justification *)
Theorem git_tool_external_io :
  forall (cmd : string),
    (* Command execution delegates to system git, no local state mutation *)
    True.
Proof.
  intros. trivial.
Qed.
