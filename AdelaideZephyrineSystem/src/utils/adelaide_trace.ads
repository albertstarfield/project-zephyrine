pragma SPARK_Mode (Off);
-- thread: Tracing requires task-safe logging

--  ============================================================================
--  ADELAIDE TRACE — Standardized verbosity for tool execution.
--  ============================================================================
--  Provides:
--    1. Server-uptime counter since Initialize().
--    2. Trace_Print with format: [prefix][Toolcall][+uptime] <message>
--    3. Trace_Result for completion status.
--    4. Prefix configured via ADELAIDE_TOOL_TRACE_PREFIX env var.
--  ============================================================================

with Ada.Real_Time;          use Ada.Real_Time;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package Adelaide_Trace is

   --  Initialize the trace system.  Call once at server start.
   procedure Initialize with Pre => True, Post => True;

   --  Whole seconds since Initialize().
   function Uptime return Natural with Pre => True, Post => True;

   --  -------------------------------------------------------------------------
   --  Trace output — emits to stdout with the standardized prefix format.
   --  -------------------------------------------------------------------------
   procedure Trace_Print (Toolcall : String; Message : String := "") with Pre => True, Post => True;
   procedure Trace_Print (Toolcall : String; Step    : String;
                          Message  : String := "") with Pre => True, Post => True;

   --  Final result trace (OK / FAIL with optional detail).
   procedure Trace_Result (Toolcall : String; Success : Boolean;
                           Detail   : String := "") with Pre => True, Post => True;

private
   Start_Time  : Ada.Real_Time.Time;
   Trace_Prefix : Unbounded_String := To_Unbounded_String ("[ADA]");
   Trace_Enabled : Boolean := True;

end Adelaide_Trace;
