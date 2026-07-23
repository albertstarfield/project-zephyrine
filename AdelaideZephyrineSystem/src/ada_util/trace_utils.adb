-- File: trace_utils.adb
-- Trace Utility Module — Standardized verbosity for all Adelaide tool scripts.

--  SPARK_Mode(off)
--  Justification: This package body performs impure I/O: reads environment
--  variables (Ada.Environment_Variables), writes diagnostics to
--  Current_Error (Ada.Text_IO), and calls Ada.Calendar.Clock for uptime.
--  These operations have side effects and cannot be verified in SPARK.
--  No security-critical or safety-critical logic is performed.

with Ada.Calendar;
with Ada.Text_IO;
with Ada.Strings;
with Ada.Strings.Fixed;
with Ada.Strings.Unbounded;
with Ada.Environment_Variables;

package body Trace_Utils is

   --  Init_Trace: Initialize tracing. Records start time, sets output
   --  prefix from argument or ADELAIDE_TOOL_TRACE_PREFIX env var.
   procedure Init_Trace (Prefix : in String := "") is
      use Ada.Strings.Unbounded;
   begin
      Start_Time := Ada.Calendar.Clock;

      if Prefix'Length > 0 then
         Trace_Utils.Prefix := To_Unbounded_String(Prefix);
      else
         --  Check environment variable
         if Ada.Environment_Variables.Exists("ADELAIDE_TOOL_TRACE_PREFIX") then
            Trace_Utils.Prefix :=
              To_Unbounded_String(
                Ada.Environment_Variables.Value("ADELAIDE_TOOL_TRACE_PREFIX"));
         else
            Trace_Utils.Prefix := To_Unbounded_String("[ADA]");
         end if;
      end if;

      --  Check if tracing is enabled
      if Ada.Environment_Variables.Exists("ADELAIDE_TOOL_TRACE_ENABLED") then
         Trace_Utils.Enabled :=
           Ada.Environment_Variables.Value("ADELAIDE_TOOL_TRACE_ENABLED") /= "0";
      end if;
   end Init_Trace;

   --  Uptime: Returns elapsed seconds since Init_Trace was called.
   function Uptime return Natural is
      use Ada.Calendar;
      Now : constant Time := Clock;
      Diff : constant Duration := Now - Start_Time;
   begin
      return Natural(Diff);
   end Uptime;

   --  Trace_Print: Emit a [prefix][Toolcall][+uptime] diagnostic line
   --  to Current_Error. Message is truncated to 200 chars.
   procedure Trace_Print
     (Toolcall : in String;
      Step     : in String := "";
      Message  : in String := "")
   is
      use Ada.Text_IO;
      use Ada.Strings.Unbounded;
      Label : Unbounded_String := To_Unbounded_String(Toolcall);
      Msg   : Unbounded_String;
   begin
      if not Enabled then
         return;
      end if;

      if Step'Length > 0 then
         Append(Label, ":" & Step);
      end if;

      --  Sanitize message: collapse whitespace, truncate to 200 chars
      if Message'Length > 0 then
         Msg := To_Unbounded_String(Message);
         if Length(Msg) > 200 then
            Msg := Head(Msg, 200);
         end if;
         Put_Line(Current_Error,
           To_String(Trace_Utils.Prefix) & "[Toolcall][+" &
           Integer'Image(Uptime) & "] " &
           To_String(Label) & ": " & To_String(Msg));
      else
         Put_Line(Current_Error,
           To_String(Trace_Utils.Prefix) & "[Toolcall][+" &
           Integer'Image(Uptime) & "] " &
           To_String(Label));
      end if;
   end Trace_Print;

   --  Trace_Result: Emit the final OK/FAIL result of a tool invocation
   --  to Current_Error with optional detail string.
   procedure Trace_Result
     (Toolcall : in String;
      Success  : in Boolean;
      Detail   : in String := "")
   is
      use Ada.Text_IO;
      use Ada.Strings.Unbounded;
      Status : constant String := (if Success then "OK" else "FAIL");
      Msg    : Unbounded_String := To_Unbounded_String(Status);
   begin
      if Detail'Length > 0 then
         Append(Msg, " -- " & Detail);
      end if;

      Put_Line(Current_Error,
        To_String(Trace_Utils.Prefix) & "[Toolcall][+" &
        Integer'Image(Uptime) & "] " &
        Toolcall & ":" & To_String(Msg));
   end Trace_Result;

end Trace_Utils;
