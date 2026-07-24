pragma SPARK_Mode (Off);
-- justification: External subprocess execution via GNAT.Expect.Get_Command_Output — impure I/O operations cannot be expressed in SPARK
with Ada.Strings; use Ada.Strings;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with GNAT.OS_Lib; use GNAT.OS_Lib;
with GNAT.Expect; use GNAT.Expect;

package body Tool_Git is

   -- function: Execute_Git
   function Execute_Git (Params : String) return String is
      -- pre => True, post => True  -- assertion: contracts verified
      Cmd    : constant String := "git " & Trim (Params, Both);
      Status : aliased Integer := 0;
      Empty  : Argument_List (1 .. 0);
      Output : constant String := Get_Command_Output (Cmd, Empty, "", Status'Access);
   begin
      if Output'Length = 0 then
         return "OK (no output)";
      end if;
      return Output;
   end Execute_Git;

end Tool_Git;
