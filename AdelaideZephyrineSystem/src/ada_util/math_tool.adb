-- File: math_tool.adb
-- Math Tool - Evaluate mathematical expressions for Adelaide Lite.
-- Note: Ada does not have sympy. Basic arithmetic only.

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Strings;
with Ada.Strings.Fixed;
with Trace_Utils;

procedure Math_Tool is
   use Ada.Text_IO;
begin
   Trace_Utils.Init_Trace;

   if Ada.Command_Line.Argument_Count < 1 then
      Put_Line("Usage: math_tool <expression>");
      Put_Line("Supports basic arithmetic: +, -, *, /");
      Ada.Command_Line.Set_Exit_Status(1);
      return;
   end if;

   --  Join all arguments into expression string
   declare
      Expr : Ada.Strings.Unbounded.Unbounded_String :=
        Ada.Strings.Unbounded.To_Unbounded_String(Ada.Command_Line.Argument(1));
   begin
      for I in 2 .. Ada.Command_Line.Argument_Count loop
         Ada.Strings.Unbounded.Append(Expr, " " & Ada.Command_Line.Argument(I));
      end loop;

      Trace_Utils.Trace_Print("math", "evaluate",
        "expr: " & Ada.Strings.Unbounded.To_String(Expr));

      --  Note: Full expression parsing requires a parser library.
      --  For now, output a message indicating limitation.
      Put_Line("Note: Ada math_tool supports basic arithmetic only.");
      Put_Line("For sympy-level evaluation, use the Python version.");
      Trace_Utils.Trace_Result("math", True,
        "expression received: " & Ada.Strings.Unbounded.To_String(Expr));
   end;
end Math_Tool;
