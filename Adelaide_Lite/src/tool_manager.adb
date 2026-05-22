with Ada.Text_IO; use Ada.Text_IO;
with GNAT.OS_Lib;
with Ada.Directories;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Model_Manager;
with Verification_Manager;
with GNATCOLL.JSON;

package body Tool_Manager is

   ------------------
   -- Execute_Tool --
   ------------------
   function Execute_Tool (Name : String; Params : String) return Tool_Result is
      Res : Tool_Result;

      --  Internal helper for executing external python tools
      function Run_External (Script : String; Arg : String) return String is
         use GNAT.OS_Lib;
         Temp_File   : constant String := "tool_cap.tmp";
         Python_Path : GNAT.OS_Lib.String_Access :=
           GNAT.OS_Lib.Locate_Exec_On_Path ("python3");
         Args        : Argument_List (1 .. 4);
         Success     : Boolean;
         Ret_Code    : Integer;
      begin
         if Python_Path = null then
            return "Error: python3 not found on system path or venv.";
         end if;

         Args (1) := new String'(Script);
         Args (2) := new String'(Arg);
         Args (3) := new String'("--ollamaHost");
         Args (4) := new String'("localhost:11420");

         Spawn (Python_Path.all, Args, Temp_File, Success, Ret_Code);
         
         Free (Python_Path);
         for I in Args'Range loop Free (Args (I)); end loop;

         if not Success then return "Error: Subprocess execution failed."; end if;

         declare
            File    : File_Type;
            Content : Unbounded_String;
            Line    : Unbounded_String;
         begin
            Open (File, In_File, Temp_File);
            while not End_Of_File (File) loop
               Line := To_Unbounded_String (Get_Line (File));
               --  STRIP BASE64 IMAGES to save context
               if Index (To_String (Line), "data:image") = 0 then
                  Append (Content, Line);
                  Append (Content, ASCII.LF);
               else
                  Append (Content, "[IMAGE STRIPPED]" & ASCII.LF);
               end if;
            end loop;
            Close (File);
            Ada.Directories.Delete_File (Temp_File);
            return To_String (Content);
         exception
            when others =>
               if Is_Open (File) then Close (File); end if;
               return "Error reading tool results.";
         end;
      end Run_External;

   begin
      Put_Line ("[Tool] Spawning native action: " & Name & " (" & Params & ")");
      Res.Success := True;

      if Name = "searchglobalref" or else Name = "search" then
         if Params = "query" then
            Res.Output := To_Unbounded_String ("No search results available.");
            return Res;
         end if;
         Res.Output := To_Unbounded_String
           (Run_External ("python/searchglobalref.py", Params));
      elsif Name = "searchlocalref" then
         Res.Output := To_Unbounded_String
           (Run_External ("python/searchlocalref.py", Params));
      elsif Name = "ls" then
         declare
            List : Unbounded_String;
            procedure Add_Item (Dir_Entry : Ada.Directories.Directory_Entry_Type) is
            begin
               Append (List, Ada.Directories.Simple_Name (Dir_Entry) & ASCII.LF);
            end Add_Item;
         begin
            Ada.Directories.Search (Params, "", (others => True), Add_Item'Access);
            Res.Output := List;
         exception
            when others =>
               Res.Success := False;
               Res.Output := To_Unbounded_String ("Error: Dir unavailable.");
         end;
      elsif Name = "cat" then
         begin
            declare
               File : File_Type;
               Content : Unbounded_String;
            begin
               Open (File, In_File, Params);
               while not End_Of_File (File) loop
                  Append (Content, Get_Line (File) & ASCII.LF);
               end loop;
               Close (File);
               Res.Output := Content;
            end;
         exception
            when others =>
               Res.Success := False;
               Res.Output := To_Unbounded_String ("Error: File unreadable.");
         end;
      elsif Name = "dafny_programmer" then
         declare
            use GNATCOLL.JSON;
            Spec : Unbounded_String := Null_Unbounded_String;
            Lang : Unbounded_String := To_Unbounded_String ("js");
            Val  : JSON_Value;
         begin
            begin
               Val := Read (Params);
               if Has_Field (Val, "specification") then
                  Spec := To_Unbounded_String (String'(Get (Val, "specification")));
               end if;
               if Has_Field (Val, "target_language") then
                  Lang := To_Unbounded_String (String'(Get (Val, "target_language")));
               end if;
            exception
               when others =>
                  -- Fallback: if Params is not valid JSON, check for a comma separator or use whole Params as spec
                  declare
                     Comma : constant Natural := Index (Params, ",");
                  begin
                     if Comma > 0 then
                        Spec := To_Unbounded_String (Trim (Params (Params'First .. Comma - 1), Ada.Strings.Both));
                        Lang := To_Unbounded_String (Trim (Params (Comma + 1 .. Params'Last), Ada.Strings.Both));
                     else
                        Spec := To_Unbounded_String (Params);
                     end if;
                  end;
            end;

            Put_Line ("[Tool] Invoking Dafny Verification & Compilation Pipeline...");
            declare
               Result_Code : constant String := Verification_Manager.Verify_And_Compile_Dafny
                 (Specification => To_String (Spec),
                  Target_Lang   => To_String (Lang),
                  Generator     => Model_Manager.Generator_Callback'Access);
            begin
               Res.Success := (Index (Result_Code, "Failed to verify Dafny code") = 0);
               Res.Output := To_Unbounded_String (Result_Code);
            end;
         end;
      else
         Res.Success := False;
         Res.Output := To_Unbounded_String ("Error: Unknown tool [" & Name & "]");
      end if;

      return Res;
   end Execute_Tool;

end Tool_Manager;
