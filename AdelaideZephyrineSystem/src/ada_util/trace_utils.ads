-- File: trace_utils.ads
-- Trace Utility Module — Standardized verbosity for all Adelaide tool scripts.
-- Provides trace_print() and trace_result() for consistent [prefix][Toolcall][+uptime] output.

with Ada.Calendar;
with Ada.Text_IO;
with Ada.Strings.Fixed;
with Ada.Strings.Unbounded;

package Trace_Utils is
   --  Initialize the trace module.  Call once at procedure start.
   procedure Init_Trace (Prefix : in String := "");

   --  Emit a [prefix][Toolcall][+uptime] trace line to Current_Error.
   procedure Trace_Print
     (Toolcall : in String;
      Step     : in String := "";
      Message  : in String := "");

   --  Trace the final result of a tool invocation.
   procedure Trace_Result
     (Toolcall : in String;
      Success  : in Boolean;
      Detail   : in String := "");

private
   Start_Time : Ada.Calendar.Time;
   Prefix     : Ada.Strings.Unbounded.Unbounded_String :=
     Ada.Strings.Unbounded.To_Unbounded_String("[ADA]");
   Enabled    : Boolean := True;
end Trace_Utils;
