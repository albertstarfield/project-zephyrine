with Interfaces.C; use Interfaces.C;
with Ada.Text_IO;  use Ada.Text_IO;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;

package body Kratos is

   function Safe_Llama_Decode
     (Context : System.Address;
      Batch   : System.Address)
      return Interfaces.C.int
   is
      function Llama_Decode_Bare
        (Ctx   : System.Address;
         Batch : System.Address)
         return Interfaces.C.int;
      pragma Import (C, Llama_Decode_Bare, "llama_decode");

      Crash_Val : Interfaces.C.int;
   begin
      Crash_Val := Guard_Enter;
      if Crash_Val = 0 then
         --  Normal path: call into llama.cpp
         declare
            R : constant Interfaces.C.int :=
              Llama_Decode_Bare (Context, Batch);
         begin
            Guard_Exit;
            return R;
         end;
      else
         --  Recovery path: signal caught during a previous Guard region
         --  This shouldn't happen on re-entry, but handle it
         Clear_Crash;
         return -1;
      end if;
   end Safe_Llama_Decode;

   procedure Log_Crash is
      Sig : constant Interfaces.C.int := Get_Crash_Signal;
       Sig_Name : constant String :=
          (case Integer (Sig) is
           when 11 => "SIGSEGV (Segmentation Fault)",
           when 7  => "SIGBUS  (Bus Error)",
           when 8  => "SIGFPE  (Floating Point Exception)",
           when 5  => "SIGTRAP (Trace/BPT Trap)",
           when 6  => "SIGABRT (Abort)",
           when others => "Signal" & Integer (Sig)'Image);
   begin
      Put_Line (Standard_Error,
        "[Kratos] *** CRASH ISOLATED *** " & Sig_Name);
      Put_Line (Standard_Error,
        "[Kratos] Server continuing - inference request aborted.");
      Clear_Crash;
   end Log_Crash;

begin
   --  Install handlers on package elaboration
   Install_Handlers;
end Kratos;
