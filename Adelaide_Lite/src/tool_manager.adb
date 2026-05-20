with Ada.Text_IO; use Ada.Text_IO;
with GNAT.OS_Lib;
with Ada.Directories;

package body Tool_Manager is

   ------------------
   -- Execute_Tool --
   ------------------
   function Execute_Tool (Name : String; Params : String) return Tool_Result is
      Res : Tool_Result;
   begin
      Put_Line ("[Tool] Adelaide executing action: " & Name & " (" & Params & ")");
      Res.Success := True;
      
      if Name = "ls" then
         declare
            Search_Pattern : constant String := (if Params = "" then "." else Params);
            List : Unbounded_String;
            
            procedure Process_Entry (Dir_Entry : Ada.Directories.Directory_Entry_Type) is
            begin
               Append (List, Ada.Directories.Simple_Name (Dir_Entry) & ASCII.LF);
            end Process_Entry;
         begin
            Ada.Directories.Search (Search_Pattern, "", (others => True), Process_Entry'Access);
            Res.Output := List;
         exception
            when others =>
               Res.Success := False;
               Res.Output := To_Unbounded_String ("Error: Directory not found.");
         end;
         
      elsif Name = "cat" then
         begin
            declare
               File : File_Type;
               Line : Unbounded_String;
            begin
               Open (File, In_File, Params);
               while not End_Of_File (File) loop
                  Append (Line, To_Unbounded_String (Get_Line (File) & ASCII.LF));
               end loop;
               Close (File);
               Res.Output := Line;
            end;
         exception
            when others =>
               Res.Success := False;
               Res.Output := To_Unbounded_String ("Error: Could not read file.");
         end;
         
      elsif Name = "search" then
         --  Dummy global search tool for the prototype
         Res.Output := To_Unbounded_String ("Found relevant info: " & Name & " is useful for " & Params);
         
      else
         Res.Success := False;
         Res.Output := To_Unbounded_String ("Error: Unknown tool " & Name);
      end if;
      
      return Res;
   end Execute_Tool;

end Tool_Manager;
