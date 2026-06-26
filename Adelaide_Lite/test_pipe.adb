with Ada.Text_IO; use Ada.Text_IO;
with GNAT.Expect; use GNAT.Expect;
procedure Test_Pipe is
   Pd : Process_Descriptor;
   Status : Integer;
   Args : GNAT.OS_Lib.Argument_List_Access := new GNAT.OS_Lib.Argument_List'(1 => new String'("-c"), 2 => new String'("while True: print(input().upper())"));
begin
   Non_Blocking_Spawn (Pd, "python3", Args.all);
   Send (Pd, "hello" & ASCII.LF);
   declare
      Res : String := Get_Command_Output (Pd, "python3", Status); -- Wait, this is blocking until process dies.
   begin
      null;
   end;
end Test_Pipe;
