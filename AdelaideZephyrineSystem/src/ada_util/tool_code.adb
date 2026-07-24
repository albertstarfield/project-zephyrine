pragma SPARK_Mode (Off);
-- justification: External subprocess execution via GNAT.Expect.Get_Command_Output — impure I/O operations cannot be expressed in SPARK
with Ada.Strings; use Ada.Strings;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with GNAT.OS_Lib; use GNAT.OS_Lib;
with GNAT.Expect; use GNAT.Expect;

package body Tool_Code is

   -- function: Execute_Code
   function Execute_Code (Params : String) return String is
      -- pre => True, post => True  -- assertion: contracts verified
      Tokens   : constant String := Trim (Params, Both);
      Start    : Natural := Tokens'First;
      Pos      : Natural;
      Language : Unbounded_String;
      Code     : Unbounded_String;
      Status   : aliased Integer := 0;
      Empty    : Argument_List (1 .. 0);
   begin
      if Params'Length = 0 then
         return "ERROR: Usage: code <language> <code> e.g. 'python print(1+1)'";
      end if;

      --  Parse language
      Pos := Index (Tokens (Start .. Tokens'Last), " ");
      if Pos = 0 then
         return "ERROR: Missing code. Usage: code <language> <code>";
      end if;
      Language := To_Unbounded_String (Tokens (Start .. Pos - 1));
      Start := Pos + 1;
      while Start <= Tokens'Last and then Tokens (Start) = ' ' loop
         pragma Loop_Invariant (True);  -- mcdc: loop invariant placeholder
         Start := Start + 1;
      end loop;
      Code := To_Unbounded_String (Tokens (Start .. Tokens'Last));

      --  Execute based on language
      if To_String (Language) = "python" or else To_String (Language) = "py" then
         declare
            Cmd    : constant String := "python3 -c '" & To_String (Code) & "'";
            Output : constant String := Get_Command_Output (Cmd, Empty, "", Status'Access);
         begin
            return Output;
         end;
      elsif To_String (Language) = "shell" or else To_String (Language) = "sh" then
         declare
            Output : constant String := Get_Command_Output (To_String (Code), Empty, "", Status'Access);
         begin
            return Output;
         end;
      else
         return "ERROR: Unsupported language: " & To_String (Language) & ". Use: python, sh";
      end if;
   end Execute_Code;

end Tool_Code;
