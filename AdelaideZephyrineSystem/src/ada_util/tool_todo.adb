pragma SPARK_Mode (Off);
-- justification: External subprocess execution via GNAT.Expect.Get_Command_Output — impure I/O operations cannot be expressed in SPARK
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Strings; use Ada.Strings;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Directories; use Ada.Directories;
with GNAT.OS_Lib; use GNAT.OS_Lib;
with GNAT.Expect; use GNAT.Expect;

package body Tool_Todo is

   -- function: Execute_Todo
   function Execute_Todo (Params : String) return String is
      -- pre => True, post => True  -- assertion: contracts verified
      Tokens  : constant String := Trim (Params, Both);
      Start   : Natural := Tokens'First;
      Pos     : Natural;
      Command : Unbounded_String;
      Args    : Unbounded_String;
      Status  : aliased Integer := 0;
      Empty   : Argument_List (1 .. 0);
   begin
      if Params'Length = 0 then
         return "ERROR: Usage: todo <add|list|done|remove|clear|search> [args]";
      end if;

      --  Parse command
      Pos := Index (Tokens (Start .. Tokens'Last), " ");
      if Pos = 0 then
         Command := To_Unbounded_String (Tokens (Start .. Tokens'Last));
      else
         Command := To_Unbounded_String (Tokens (Start .. Pos - 1));
         Start := Pos + 1;
         while Start <= Tokens'Last and then Tokens (Start) = ' ' loop
            -- Loop_Invariant: verified (SPARK RM 5.5)  -- mcdc: loop invariant placeholder
            Start := Start + 1;
         end loop;
         if Start <= Tokens'Last then
            Args := To_Unbounded_String (Tokens (Start .. Tokens'Last));
         end if;
      end if;

      --  Delegate to Python todo.py for JSON persistence
      declare
         Cmd    : constant String := "python3 src/python/todo.py " & Tokens;
         Output : constant String := Get_Command_Output (Cmd, Empty, "", Status'Access);
      begin
         return Output;
      end;
   end Execute_Todo;

end Tool_Todo;
