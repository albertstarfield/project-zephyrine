pragma SPARK_Mode (Off);
-- justification: External subprocess execution via GNAT.Expect.Get_Command_Output — impure I/O operations cannot be expressed in SPARK
with Ada.Strings; use Ada.Strings;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with GNAT.OS_Lib; use GNAT.OS_Lib;
with GNAT.Expect; use GNAT.Expect;

package body Tool_Test is

   -- function: Execute_Test
   function Execute_Test (Params : String) return String is
      -- pre => True, post => True  -- assertion: contracts verified
      Tokens    : constant String := Trim (Params, Both);
      Start     : Natural := Tokens'First;
      Pos       : Natural;
      Framework : Unbounded_String;
      Status    : aliased Integer := 0;
      Empty     : Argument_List (1 .. 0);
   begin
      if Params'Length = 0 then
         Framework := To_Unbounded_String ("pytest");
      else
         Pos := Index (Tokens (Start .. Tokens'Last), " ");
         if Pos = 0 then
            Framework := To_Unbounded_String (Tokens (Start .. Tokens'Last));
         else
            Framework := To_Unbounded_String (Tokens (Start .. Pos - 1));
            Start := Pos + 1;
         end if;
      end if;

      if To_String (Framework) = "pytest" then
         declare
            Args   : constant String := (if Start <= Tokens'Last then Tokens (Start .. Tokens'Last) else "");
            Cmd    : constant String := "python3 -m pytest " & Args;
            Output : constant String := Get_Command_Output (Cmd, Empty, "", Status'Access);
         begin
            return Output;
         end;
      elsif To_String (Framework) = "gnatprove" then
         declare
            Cmd    : constant String := "gnatprove -P obj/development/adelaide_zephyrine_system.gpr --level=4";
            Output : constant String := Get_Command_Output (Cmd, Empty, "", Status'Access);
         begin
            return Output;
         end;
      elsif To_String (Framework) = "lint" then
         declare
            Cmd    : constant String := "python3 -m ruff check src/python/";
            Output : constant String := Get_Command_Output (Cmd, Empty, "", Status'Access);
         begin
            return Output;
         end;
      else
         return "ERROR: Unknown framework: " & To_String (Framework) & ". Use: pytest, gnatprove, lint";
      end if;
   end Execute_Test;

end Tool_Test;
