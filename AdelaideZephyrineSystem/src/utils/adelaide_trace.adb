pragma SPARK_Mode (Off);
-- thread: Tracing requires task-safe logging

--  ============================================================================
--  ADELAIDE TRACE — Implementation
--  ============================================================================

with Ada.Text_IO;  use Ada.Text_IO;
with Ada.Calendar; use Ada.Calendar;
with GNAT.OS_Lib;

package body Adelaide_Trace is

   --  ------------------------------------------------------------------------
   --  Initialize
   --  ------------------------------------------------------------------------
   procedure Initialize is
      use GNAT.OS_Lib;
      Env_Val  : GNAT.OS_Lib.String_Access;
      Env_Flag : GNAT.OS_Lib.String_Access;
   begin
      Start_Time := Ada.Calendar.Clock;

      --  Read prefix from environment
      Env_Val := Getenv ("ADELAIDE_TOOL_TRACE_PREFIX");
      if Env_Val /= null and then Env_Val.all'Length > 0 then
         Trace_Prefix := To_Unbounded_String (Env_Val.all);
      end if;
      Free (Env_Val);

      --  Disable traces if ADELAIDE_TOOL_TRACE_ENABLED = "0"
      Env_Flag := Getenv ("ADELAIDE_TOOL_TRACE_ENABLED");
      if Env_Flag /= null and then Env_Flag.all = "0" then
         Trace_Enabled := False;
      end if;
      Free (Env_Flag);
   end Initialize;

   --  ------------------------------------------------------------------------
   --  Uptime
   --  ------------------------------------------------------------------------
   function Uptime return Natural is
   begin
      return Natural ( Ada.Calendar."-" (Ada.Calendar.Clock, Start_Time) );
   end Uptime;

   --  ------------------------------------------------------------------------
   --  Trace_Print (two-argument form)
   --  ------------------------------------------------------------------------
   procedure Trace_Print (Toolcall : String; Message : String := "") is
   begin
      if not Trace_Enabled then
         return;
      end if;
      declare
         U   : constant Natural := Uptime;
         Msg : constant String :=
           To_String (Trace_Prefix) & "[Toolcall][+" &
           Natural'Image (U)(2 .. Natural'Image (U)'Last) & "] " &
           Toolcall;
      begin
         if Message'Length > 0 then
            Put_Line (Msg & ": " & Message);
         else
            Put_Line (Msg);
         end if;
      end;
   end Trace_Print;

   --  ------------------------------------------------------------------------
   --  Trace_Print (three-argument form with Step)
   --  ------------------------------------------------------------------------
   procedure Trace_Print (Toolcall : String; Step    : String;
                          Message  : String := "") is
   begin
      if not Trace_Enabled then
         return;
      end if;
      declare
         U          : constant Natural := Uptime;
         Label      : constant String := Toolcall & ":" & Step;
         Prefix     : constant String := To_String (Trace_Prefix);
         Time_Stamp : constant String :=
           Natural'Image (U)(2 .. Natural'Image (U)'Last);
         Full       : constant String :=
           Prefix & "[Toolcall][+" & Time_Stamp & "] " & Label;
      begin
         if Message'Length > 0 then
            Put_Line (Full & ": " & Message);
         else
            Put_Line (Full);
         end if;
      end;
   end Trace_Print;

   --  ------------------------------------------------------------------------
   --  Trace_Result
   --  ------------------------------------------------------------------------
   procedure Trace_Result (Toolcall : String; Success : Boolean;
                           Detail   : String := "") is
      Status : constant String := (if Success then "OK" else "FAIL");
   begin
      if not Trace_Enabled then
         return;
      end if;
      declare
         U   : constant Natural := Uptime;
         Msg : constant String :=
           To_String (Trace_Prefix) & "[Toolcall][+" &
           Natural'Image (U)(2 .. Natural'Image (U)'Last) & "] " &
           Toolcall & ":" & Status;
      begin
         if Detail'Length > 0 then
            Put_Line (Msg & " - " & Detail);
         else
            Put_Line (Msg);
         end if;
      end;
   end Trace_Result;

end Adelaide_Trace;
