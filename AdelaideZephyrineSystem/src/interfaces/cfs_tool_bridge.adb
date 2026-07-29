pragma SPARK_Mode (Off);
--  cFS Tool Bridge — ELP0/ELP1 (LLM cognitive layer) → cFS Software Bus
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Strings; use Ada.Strings;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with CFE_FFI_Bindings;
with CFS_Telemetry;
with CFS_Health_Monitor;
with CFS_Command_Router;

package body CFS_Tool_Bridge is

   --  Extract first word from params (subcommand)
   function Get_Subcommand (Params : String) return String is
      Sp : Natural := Index (Params, " ");
   begin
      if Sp = 0 then
         return Params;
      else
         return Params (Params'First .. Sp - 1);
      end if;
   end Get_Subcommand;

   --  Extract remainder after first word
   function Get_Rest (Params : String) return String is
      Sp : Natural := Index (Params, " ");
   begin
      if Sp = 0 then
         return "";
      else
         return Params (Sp + 1 .. Params'Last);
      end if;
   end Get_Rest;

   --  ──────────────────────────────────────────────────────────────────────
   --  Execute_CFS_Tool — main dispatcher
   --  ──────────────────────────────────────────────────────────────────────
   function Execute_CFS_Tool (Params : String) return Tool_Result is
      Sub : constant String := Get_Subcommand (Params);
      Rest : constant String := Get_Rest (Params);
   begin
      --  Ensure cFS is initialized
      CFE_FFI_Bindings.CFE_Initialize;
      CFS_Telemetry.Initialize;
      CFS_Health_Monitor.Initialize;
      CFS_Command_Router.Initialize;

      if Sub = "status" then
         --  cfs status — overall system status
         declare
            Sys_Health : CFS_Health_Monitor.Health_Status :=
              CFS_Health_Monitor.Get_System_Health;
            Cmd_Count : constant Natural :=
              CFS_Command_Router.Get_Command_Count;
         begin
            return (Success => True,
                    Output  =>
                      To_Unbounded_String (
                        "[cFS Status]" & ASCII.LF &
                        "  Health:    " & CFS_Health_Monitor.Health_Status'Image (Sys_Health) & ASCII.LF &
                        "  Commands:  " & Natural'Image (Cmd_Count) & " routed" & ASCII.LF &
                        "  Telemetry: Active" & ASCII.LF &
                        "  SW Bus:    Initialized"));
         end;

      elsif Sub = "telemetry" or else Sub = "tlm" then
         --  cfs telemetry <type> [args]
         declare
            Tlm_Type : constant String := Get_Subcommand (Rest);
         begin
            if Tlm_Type = "hk" or else Tlm_Type = "housekeeping" then
               CFS_Telemetry.Send_Housekeeping (CPU_Pct => 0.0, Mem_Pct => 0.0, Uptime => 0.0);
               return (Success => True,
                       Output  => To_Unbounded_String ("[cFS] Housekeeping telemetry sent"));
            elsif Tlm_Type = "sensor" then
               CFS_Telemetry.Send_Sensor_Telemetry ("generic", 0.0);
               return (Success => True,
                       Output  => To_Unbounded_String ("[cFS] Sensor telemetry sent"));
            elsif Tlm_Type = "attitude" or else Tlm_Type = "att" then
               CFS_Telemetry.Send_Attitude_Telemetry (0.0, 0.0, 0.0);
               return (Success => True,
                       Output  => To_Unbounded_String ("[cFS] Attitude telemetry sent"));
            else
               return (Success => False,
                       Output  => To_Unbounded_String ("[cFS] Unknown telemetry type: " & Tlm_Type &
                         ". Use: hk, sensor, attitude"));
            end if;
         end;

      elsif Sub = "health" then
         --  cfs health [app_name]
         if Rest'Length = 0 then
            declare
               Sys : constant CFS_Health_Monitor.Health_Status :=
                 CFS_Health_Monitor.Get_System_Health;
            begin
               return (Success => True,
                       Output  => To_Unbounded_String (
                         "[cFS] System Health: " &
                         CFS_Health_Monitor.Health_Status'Image (Sys)));
            end;
         else
            declare
               App_H : constant CFS_Health_Monitor.Health_Status :=
                 CFS_Health_Monitor.Check_App_Health (Rest);
            begin
               return (Success => True,
                       Output  => To_Unbounded_String (
                         "[cFS] App '" & Rest & "' Health: " &
                         CFS_Health_Monitor.Health_Status'Image (App_H)));
            end;
         end if;

      elsif Sub = "command" or else Sub = "cmd" then
         --  cfs command <type> <data>
         declare
            Cmd_Type_Str : constant String := Get_Subcommand (Rest);
            Cmd_Data     : constant String := Get_Rest (Rest);
            Cmd_T        : CFS_Command_Router.Cmd_Type;
         begin
            if Cmd_Type_Str = "gnc" then
               Cmd_T := CFS_Command_Router.GNC;
            elsif Cmd_Type_Str = "telemetry" or else Cmd_Type_Str = "tlm" then
               Cmd_T := CFS_Command_Router.Telemetry;
            elsif Cmd_Type_Str = "health" then
               Cmd_T := CFS_Command_Router.Health;
            elsif Cmd_Type_Str = "config" then
               Cmd_T := CFS_Command_Router.Configuration;
            else
               Cmd_T := CFS_Command_Router.Custom;
            end if;

            declare
               Cmd : CFS_Command_Router.Command;
            begin
               Cmd.Cmd_Type := Cmd_T;
               Cmd.Cmd_Len := Cmd_Data'Length;
               Cmd.Cmd_Data (1 .. Cmd_Data'Length) := Cmd_Data;
               CFS_Command_Router.Route_Command (Cmd);
            end;

            return (Success => True,
                    Output  => To_Unbounded_String (
                      "[cFS] Command routed: " & Cmd_Type_Str));
         end;

      elsif Sub = "info" then
         --  cfs info — version and configuration
         return (Success => True,
                 Output  =>
                   To_Unbounded_String (
                     "[cFS Info]" & ASCII.LF &
                     "  Framework: NASA core Flight System (cFS)" & ASCII.LF &
                     "  Version:   7.0.1 (Draco)" & ASCII.LF &
                     "  License:   Apache 2.0" & ASCII.LF &
                     "  Components:" & ASCII.LF &
                     "    cFE:  Core Flight Executive" & ASCII.LF &
                     "    OSAL: OS Abstraction Layer" & ASCII.LF &
                     "    PSP:  Platform Support Package" & ASCII.LF &
                     "    Apps: CI_LAB, TO_LAB, SCH_LAB, HS, FM, DS, etc." & ASCII.LF &
                     "  Integration: Ada FFI via Interfaces.C"));

      else
         return (Success => False,
                 Output  => To_Unbounded_String (
                   "[cFS] Unknown subcommand: " & Sub & ASCII.LF &
                   "Available: status, telemetry, health, command, info"));
      end if;
   end Execute_CFS_Tool;

end CFS_Tool_Bridge;
