pragma SPARK_Mode (Off);
with AnsiAda;
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with GNAT.OS_Lib;
with GNAT.Expect;

package body Tool_Manager is

   function Execute_Tool (Name : String; Params : String) return Tool_Result is
      use GNAT.OS_Lib;
      Path : GNAT.OS_Lib.String_Access;
      Full_Cmd : Unbounded_String;
      Result : Tool_Result := (Success => False,
                               Output  => Null_Unbounded_String);
   begin
      Path := GNAT.OS_Lib.Locate_Exec_On_Path ("python3");
      if Path = null then
         Result.Output := To_Unbounded_String ("Error: python3 not found");
         return Result;
      end if;

      Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[Tool]" & AnsiAda.Reset &
                " Executing: " & Name & " with params: " & Params);

      if Name = "web_search" or else Name = "searchglobalref" or else Name = "search" then
         Full_Cmd := To_Unbounded_String ("python/searchglobalref.py");
      elsif Name = "local_search" then
         Full_Cmd := To_Unbounded_String ("python/searchlocalref.py");
      elsif Name = "math" then
         Full_Cmd := To_Unbounded_String ("python/math_tool.py");
      elsif Name = "code" then
         Full_Cmd := To_Unbounded_String ("python/code_tool.py");
      elsif Name = "cat" then
         Full_Cmd := To_Unbounded_String ("python/cat_tool.py");
      else
         Free (Path);
         Result.Output := To_Unbounded_String ("Error: Unknown tool " & Name);
         return Result;
      end if;

      declare
         Arg_List : Argument_List (1 .. 2);
         Status : aliased Integer;
      begin
         Arg_List (1) := new String'(To_String (Full_Cmd));
         Arg_List (2) := new String'(Params);
         
         Result.Output := To_Unbounded_String
           (GNAT.Expect.Get_Command_Output (Path.all, Arg_List, "",
            Status'Access));

         for I in Arg_List'Range loop Free (Arg_List (I)); end loop;
         Free (Path);
         
         Result.Success := True;
         return Result;
      end;
   exception
      when others =>
         if Path /= null then Free (Path); end if;
         Result.Output := To_Unbounded_String ("Error executing tool");
         return Result;
   end Execute_Tool;

end Tool_Manager;
