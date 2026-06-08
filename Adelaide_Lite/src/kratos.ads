pragma SPARK_Mode (Off);
--  Kratos Crash Isolation Layer
--
--  Protects inference threads from C-level crashes (SIGSEGV, SIGBUS, SIGFPE).
--  If llama.cpp faults during Llama_Decode or similar, Kratos catches the
--  signal and longjmps back to a safe recovery point instead of killing
--  the entire Ada server process.
--
--  Usage:
--    if Kratos.Guard_Enter = 0 then
--       --  Normal path: call C FFI here
--       Result := Llama_Decode (Ctx, Batch);
--       Kratos.Guard_Exit;
--    else
--       --  Recovery path: signal was caught
--       Kratos.Log_Crash;
--       Result := -1;
--    end if;

with Interfaces.C;
with System;

package Kratos is

   --  Install signal handlers (SIGSEGV, SIGBUS, SIGFPE).
   --  Safe to call multiple times (idempotent).
   procedure Install_Handlers;
   pragma Import (C, Install_Handlers, "jorvik_install_handlers");

   --  Enter a protected region. Returns 0 on normal entry.
   --  Returns nonzero (the signal number) if recovering from a crash.
   function Guard_Enter return Interfaces.C.int;
   pragma Import (C, Guard_Enter, "jorvik_guard_enter");

   --  Exit a protected region. Must be called on the normal path.
   procedure Guard_Exit;
   pragma Import (C, Guard_Exit, "jorvik_guard_exit");

   --  Check if a crash occurred (nonzero = yes).
   function Crash_Occurred return Interfaces.C.int;
   pragma Import (C, Crash_Occurred, "jorvik_crash_occurred");

   --  Get the signal number that caused the crash.
   function Get_Crash_Signal return Interfaces.C.int;
   pragma Import (C, Get_Crash_Signal, "jorvik_get_crash_signal");

   --  Clear crash state after recovery.
   procedure Clear_Crash;
   pragma Import (C, Clear_Crash, "jorvik_clear_crash");

   --  Ada-friendly wrappers
   function Safe_Llama_Decode
     (Context : System.Address;
      Batch   : System.Address)
      return Interfaces.C.int;

   --  Log crash details to stderr
   procedure Log_Crash;

end Kratos;
