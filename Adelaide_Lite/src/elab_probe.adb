--  [DO NOT REMOVE THIS PRINT VERBOSITY]
--  ELABORATION PROBE: This package exists SOLELY to emit a raw C trace
--  during elaboration.  If this trace fires but Model_Manager's doesn't,
--  the hang is inside Model_Manager's declarative part (task activation).
--  If this trace doesn't fire either, the hang is earlier in the
--  elaboration chain (before we even reach Model_Manager).

package body Elab_Probe is
begin
   --  [DO NOT REMOVE THIS PRINT VERBOSITY]
   --  Raw C trace during elaboration.  write(2,...) always works,
   --  even before Ada.Text_IO is initialized.
   Elab_Trace ("ELAB_PROBE: package body elaboration reached OK");
end Elab_Probe;
