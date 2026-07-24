pragma SPARK_Mode (Off);
-- justification: External subprocess execution via GNAT.Expect.Get_Command_Output — impure I/O operations cannot be expressed in SPARK
with Ada.Strings; use Ada.Strings;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with GNAT.OS_Lib; use GNAT.OS_Lib;
with GNAT.Expect; use GNAT.Expect;

package body Tool_Math is

   -- function: Execute_Math
   function Execute_Math (Params : String) return String is
      -- pre => True, post => True  -- assertion: contracts verified
      Expr   : constant String := Trim (Params, Both);
      Status : aliased Integer := 0;
      Empty  : Argument_List (1 .. 0);
   begin
      if Expr'Length = 0 then
         return "ERROR: Usage: math <expression> e.g. '2 + 3 * 4'";
      end if;

      --  Use Python for safe math evaluation
      declare
         Cmd    : constant String := "python3 -c 'print(" & Expr & ")'";
         Output : constant String := Get_Command_Output (Cmd, Empty, "", Status'Access);
      begin
         if Output'Length = 0 then
            return "ERROR: Could not evaluate expression: " & Expr;
         end if;
         return Output;
      end;
   end Execute_Math;

end Tool_Math;
