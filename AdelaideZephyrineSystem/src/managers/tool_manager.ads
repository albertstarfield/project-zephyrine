pragma SPARK_Mode (Off);
-- thread: Tool execution requires task protection
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package Tool_Manager is

   type Tool_Result is record
      Success : Boolean;
      Output  : Unbounded_String;
   end record;

   -- function: Execute_Tool — route named tool to native Ada implementation
   function Execute_Tool (Name : String; Params : String) return Tool_Result
     with Pre => True, Post => True;

   -- function: Execute_Imagine_Tool — image generation via SD_Manager
   function Execute_Imagine_Tool (Prompt : String) return Tool_Result
     with Pre => True, Post => True;

   -- function: Execute_Cronia_Tool — timed answer on ELP0
   function Execute_Cronia_Tool (Params : String) return Tool_Result
     with Pre => True, Post => True;

   -- function: Execute_Proactive_Tool — proactive question or handless mode
   function Execute_Proactive_Tool (Params : String) return Tool_Result
     with Pre => True, Post => True;

   -- function: Execute_ROS2_Tool — native Ada ROS2 actuator via ELP3
   function Execute_ROS2_Tool (Params : String) return Tool_Result
     with Pre => True, Post => True;

   -- Native Ada tools (replacing Python subprocess spawning)
   -- function: Execute_Cat — read and display file contents
   function Execute_Cat (Params : String) return Tool_Result
     with Pre => True, Post => True;
   -- function: Execute_Grep — search file contents by pattern
   function Execute_Grep (Params : String) return Tool_Result
     with Pre => True, Post => True;
   -- function: Execute_Git — execute git commands
   function Execute_Git (Params : String) return Tool_Result
     with Pre => True, Post => True;
   -- function: Execute_File_Edit — create, append, write, or delete files
   function Execute_File_Edit (Params : String) return Tool_Result
     with Pre => True, Post => True;
   -- function: Execute_Dir — list, find, tree, pwd, mkdir, rm
   function Execute_Dir (Params : String) return Tool_Result
     with Pre => True, Post => True;
   -- function: Execute_Todo — task management via JSON persistence
   function Execute_Todo (Params : String) return Tool_Result
     with Pre => True, Post => True;
   -- function: Execute_Killshell — process management (kill, list, find)
   function Execute_Killshell (Params : String) return Tool_Result
     with Pre => True, Post => True;
   -- function: Execute_Math — evaluate math expressions via Python
   function Execute_Math (Params : String) return Tool_Result
     with Pre => True, Post => True;
   -- function: Execute_Code — execute code snippets (python, shell)
   function Execute_Code (Params : String) return Tool_Result
     with Pre => True, Post => True;
   -- function: Execute_Test — run test frameworks (pytest, gnatprove, lint)
   function Execute_Test (Params : String) return Tool_Result
     with Pre => True, Post => True;
   -- function: Execute_Issue — GitHub issue management
   function Execute_Issue (Params : String) return Tool_Result
     with Pre => True, Post => True;
   -- function: Execute_Review — code review (diff, pr)
   function Execute_Review (Params : String) return Tool_Result
     with Pre => True, Post => True;
   -- function: Execute_Hook — git hook management
   function Execute_Hook (Params : String) return Tool_Result
     with Pre => True, Post => True;
   -- function: Execute_Package — package manager (brew, apt)
   function Execute_Package (Params : String) return Tool_Result
     with Pre => True, Post => True;
   -- function: Execute_CFS_Tool — NASA cFS flight software (telemetry, health, commands)
   function Execute_CFS_Tool (Params : String) return Tool_Result
     with Pre => True, Post => True;

end Tool_Manager;
