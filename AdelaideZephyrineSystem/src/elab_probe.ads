--  [DO NOT REMOVE THIS PRINT VERBOSITY]
--  ELABORATION PROBE SPEC: Declares a C FFI trace function and calls it
--  during package body elaboration.  If this trace fires, we know the
--  elaboration chain reached this package.  If it doesn't fire, the
--  hang is earlier in the elaboration chain.

package Elab_Probe is
   pragma Elaborate_Body;
   --  [DO NOT REMOVE THIS PRINT VERBOSITY]
   --  C FFI: Raw write to stderr, bypasses all buffering.
   --  ABI NOTE: GNAT passes String as fat pointer (data_ptr, bounds_ptr).
   --  C side uses strlen() to measure the string — do NOT pass a length.
   procedure Elab_Trace (Label : String);
   pragma Import (C, Elab_Trace, "elab_trace_c");
end Elab_Probe;
