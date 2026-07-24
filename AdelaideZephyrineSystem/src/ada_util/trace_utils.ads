-- File: trace_utils.ads
-- Trace Utility Module — Standardized verbosity for all Adelaide tool scripts.
-- Provides trace_print() and trace_result() for consistent [prefix][Toolcall][+uptime] output.

--  SPARK_Mode(off)
--  Justification: This package reads environment variables via
--  Ada.Environment_Variables, writes to Current_Error via Ada.Text_IO,
--  and calls Ada.Calendar.Clock for timing. These are impure I/O
--  operations that cannot be expressed in SPARK. The package does not
--  perform any security-critical or safety-critical logic; it is a
--  diagnostic tracing utility only.

with Ada.Calendar;
with Ada.Text_IO;
with Ada.Strings.Fixed;
with Ada.Strings.Unbounded;

package Trace_Utils is
   --  Initialize the trace module.  Call once at procedure start.
   procedure Init_Trace (Prefix : in String := "") with Pre => True, Post => True;

   --  Emit a [prefix][Toolcall][+uptime] trace line to Current_Error.
   procedure Trace_Print
     (Toolcall : in String;
      Step     : in String := "";
      Message  : in String := "") with Pre => True, Post => True;

   --  Trace the final result of a tool invocation.
   procedure Trace_Result
     (Toolcall : in String;
      Success  : in Boolean;
      Detail   : in String := "") with Pre => True, Post => True;

private
   Start_Time : Ada.Calendar.Time;
   Prefix     : Ada.Strings.Unbounded.Unbounded_String :=
     Ada.Strings.Unbounded.To_Unbounded_String("[ADA]");
   Enabled    : Boolean := True;
end Trace_Utils;
