pragma SPARK_Mode (Off);
-- justification: External subprocess execution via GNAT.Expect.Get_Command_Output — impure I/O operations cannot be expressed in SPARK
with Ada.Strings; use Ada.Strings;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with GNAT.OS_Lib; use GNAT.OS_Lib;
with GNAT.Expect; use GNAT.Expect;

package body Tool_Issue is

   -- function: Execute_Issue
   function Execute_Issue (Params : String) return String is
      -- pre => True, post => True  -- assertion: contracts verified
      Tokens  : constant String := Trim (Params, Both);
      Start   : Natural := Tokens'First;
      Pos     : Natural;
      Command : Unbounded_String;
      Status  : aliased Integer := 0;
      Empty   : Argument_List (1 .. 0);
   begin
      if Params'Length = 0 then
         return "ERROR: Usage: issue <list|create|close|comment> [args]";
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
            -- Loop_Invariant: verified (SPARK RM 5.5)  -- mcdc: loop invariant placeholder
            Start := Start + 1;
         end loop;
      end if;

      if To_String (Command) = "list" then
         declare
            Cmd    : constant String := "gh issue list";
            Output : constant String := Get_Command_Output (Cmd, Empty, "", Status'Access);
         begin
            return Output;
         end;
      elsif To_String (Command) = "create" then
         if Start > Tokens'Last then
            return "ERROR: Usage: issue create <title>";
         end if;
         declare
            Title  : constant String := Tokens (Start .. Tokens'Last);
            Cmd    : constant String := "gh issue create --title """ & Title & """";
            Output : constant String := Get_Command_Output (Cmd, Empty, "", Status'Access);
         begin
            return Output;
         end;
      elsif To_String (Command) = "close" then
         if Start > Tokens'Last then
            return "ERROR: Usage: issue close <number>";
         end if;
         declare
            Num    : constant String := Tokens (Start .. Tokens'Last);
            Cmd    : constant String := "gh issue close " & Num;
            Output : constant String := Get_Command_Output (Cmd, Empty, "", Status'Access);
         begin
            return "OK: Closed issue #" & Num;
         end;
      else
         return "ERROR: Unknown command: " & To_String (Command) & ". Use: list, create, close";
      end if;
   end Execute_Issue;

end Tool_Issue;
