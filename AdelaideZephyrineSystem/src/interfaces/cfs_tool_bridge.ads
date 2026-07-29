pragma SPARK_Mode (Off);
--  cFS Tool Bridge — ELP0/ELP1 (LLM cognitive layer) → cFS Software Bus
--  Provides tool-call bindings so the LLM can:
--   - Query telemetry (cfs_tlm)
--   - Send commands (cfs_cmd)
--   - Check health (cfs_health)
--   - Get system status (cfs_status)
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package CFS_Tool_Bridge is

   type Tool_Result is record
      Success : Boolean;
      Output  : Unbounded_String;
   end record;

   --  Main cFS tool dispatcher — routes tool calls to cFS subsystems
   --  Params format: "<subcommand> [args...]"
   --    "status"               — overall cFS status
   --    "telemetry <type>"     — send telemetry (hk, sensor, attitude)
   --    "health [app_name]"    — check health of app or system
   --    "command <type> <data>" — send command through Software Bus
   --    "info"                 — cFS version and config info
   function Execute_CFS_Tool (Params : String) return Tool_Result
     with Pre => Params'Length > 0;

end CFS_Tool_Bridge;
