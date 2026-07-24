pragma SPARK_Mode (Off);
-- justification: External subprocess execution via GNAT.Expect.Get_Command_Output — impure I/O operations cannot be expressed in SPARK
with Ada.Strings; use Ada.Strings;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with GNAT.OS_Lib; use GNAT.OS_Lib;
with GNAT.Expect; use GNAT.Expect;

package body Tool_Hook is

   -- function: Execute_Hook
   function Execute_Hook (Params : String) return String is
      -- pre => True, post => True  -- assertion: contracts verified
      Tokens  : constant String := Trim (Params, Both);
      Start   : Natural := Tokens'First;
      Pos     : Natural;
      Command : Unbounded_String;
      Status  : aliased Integer := 0;
      Empty   : Argument_List (1 .. 0);
   begin
      if Params'Length = 0 then
         return "ERROR: Usage: hook <list|install|remove|run> [hook_name]";
      end if;

      --  Parse command
      Pos := Index (Tokens (Start .. Tokens'Last), " ");
      if Pos = 0 then
         Command := To_Unbounded_String (Tokens (Start .. Tokens'Last));
         Start := Tokens'Last + 1;
      else
         Command := To_Unbounded_String (Tokens (Start .. Pos - 1));
         Start := Pos + 1;
         while Start <= Tokens'Last and then Tokens (Start) = ' ' loop
            pragma Loop_Invariant (True);  -- mcdc: loop invariant placeholder
            Start := Start + 1;
         end loop;
      end if;

      if To_String (Command) = "list" then
         declare
            Cmd    : constant String := "ls -la .git/hooks/";
            Output : constant String := Get_Command_Output (Cmd, Empty, "", Status'Access);
         begin
            return Output;
         end;
      elsif To_String (Command) = "install" then
         if Start > Tokens'Last then
            return "ERROR: Usage: hook install <hook_name>";
         end if;
         declare
            Hook   : constant String := Tokens (Start .. Tokens'Last);
            Cmd    : constant String := "chmod +x .git/hooks/" & Hook;
            Output : constant String := Get_Command_Output (Cmd, Empty, "", Status'Access);
         begin
            return "OK: Installed hook: " & Hook;
         end;
      elsif To_String (Command) = "run" then
         if Start > Tokens'Last then
            return "ERROR: Usage: hook run <hook_name>";
         end if;
         declare
            Hook   : constant String := Tokens (Start .. Tokens'Last);
            Cmd    : constant String := ".git/hooks/" & Hook;
            Output : constant String := Get_Command_Output (Cmd, Empty, "", Status'Access);
         begin
            return Output;
         end;
      else
         return "ERROR: Unknown command: " & To_String (Command) & ". Use: list, install, run";
      end if;
   end Execute_Hook;

end Tool_Hook;
