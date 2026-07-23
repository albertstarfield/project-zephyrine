-- File: review_tool.adb
-- Review Tool - Code review for Adelaide Lite.

--  SPARK_Mode(off)
--  Justification: Standalone CLI procedure. Executes external processes
--  via Ada.Processes.Command_Line (git diff), reads files via
--  Ada.Text_IO and Ada.Directories, searches strings via
--  Ada.Strings.Fixed, accesses command-line arguments via
--  Ada.Command_Line. External subprocess and filesystem interaction
--  cannot be expressed in SPARK.

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Strings;
with Ada.Strings.Unbounded;
with Ada.Processes;
with Ada.Directories;
with Ada.Strings.Fixed;
with Trace_Utils;

--  Review_Tool: Main entry point. Dispatches code review commands
--  (diff, file, security, quality) for codebase inspection.
procedure Review_Tool is
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

   --  Run_Command: Execute a shell command via subprocess.
   function Run_Command (Cmd : in String) return String is
   begin
      begin
         Ada.Processes.Command_Line(
           Command_Line => Cmd,
           Output       => True);
         return "";
      exception
         when others =>
            return "";
      end;
   end Run_Command;

   --  Security_Check: Scan file for dangerous patterns (eval, exec,
   --  shell=True, pickle, os.system) and report findings.
   procedure Security_Check (Filepath : in String) is
   begin
      if not Ada.Directories.Exists(Filepath) then
         Put_Line("ERROR: File not found: " & Filepath);
         return;
      end if;

      declare
         File : File_Type;
         Line_Num : Natural := 0;
      begin
         Open(File, In_File, Filepath);
         while not End_Of_File(File) loop
            declare
               Line : constant String := Get_Line(File);
            begin
               Line_Num := Line_Num + 1;

               --  Check for security patterns
               if Ada.Strings.Fixed.Index(Line, "eval(") > 0 then
                  Put_Line("Line" & Natural'Image(Line_Num) &
                    ": Use of eval() - potential code injection");
               end if;
               if Ada.Strings.Fixed.Index(Line, "exec(") > 0 then
                  Put_Line("Line" & Natural'Image(Line_Num) &
                    ": Use of exec() - potential code injection");
               end if;
               if Ada.Strings.Fixed.Index(Line, "os.system(") > 0 then
                  Put_Line("Line" & Natural'Image(Line_Num) &
                    ": Use of os.system() - use subprocess instead");
               end if;
               if Ada.Strings.Fixed.Index(Line, "shell=True") > 0 then
                  Put_Line("Line" & Natural'Image(Line_Num) &
                    ": shell=True in subprocess - command injection risk");
               end if;
               if Ada.Strings.Fixed.Index(Line, "pickle.loads(") > 0 then
                  Put_Line("Line" & Natural'Image(Line_Num) &
                    ": Untrusted pickle deserialization");
               end if;
            end;
         end loop;
         Close(File);
      end;
   end Security_Check;

   --  Quality_Check: Scan file for quality issues (long lines,
   --  TODO/FIXME markers) and report findings.
   procedure Quality_Check (Filepath : in String) is
   begin
      if not Ada.Directories.Exists(Filepath) then
         Put_Line("ERROR: File not found: " & Filepath);
         return;
      end if;

      declare
         File : File_Type;
         Line_Num : Natural := 0;
      begin
         Open(File, In_File, Filepath);
         while not End_Of_File(File) loop
            declare
               Line : constant String := Get_Line(File);
            begin
               Line_Num := Line_Num + 1;

               --  Long lines
               if Line'Length > 120 then
                  Put_Line("Line" & Natural'Image(Line_Num) &
                    ": Line too long (" & Natural'Image(Line'Length) & " > 120)");
               end if;

               --  TODO/FIXME
               if Ada.Strings.Fixed.Index(Line, "TODO") > 0 or
                  Ada.Strings.Fixed.Index(Line, "FIXME") > 0
               then
                  Put_Line("Line" & Natural'Image(Line_Num) &
                    ": Unresolved TODO/FIXME");
               end if;
            end;
         end loop;
         Close(File);
      end;
   end Quality_Check;

begin
   Trace_Utils.Init_Trace;

   if Ada.Command_Line.Argument_Count < 1 then
      Put_Line("Usage: review_tool <command> [args...]");
      Put_Line("Commands: diff, file, security, quality");
      Ada.Command_Line.Set_Exit_Status(1);
      return;
   end if;

   declare
      Cmd  : constant String := Ada.Command_Line.Argument(1);
      Args : Unbounded_String := Null_Unbounded_String;
   begin
      for I in 2 .. Ada.Command_Line.Argument_Count loop
         if I > 2 then
            Append(Args, " ");
         end if;
         Append(Args, Ada.Command_Line.Argument(I));
      end loop;

      if Cmd = "diff" then
         declare
            Branch : constant String :=
              (if Ada.Command_Line.Argument_Count >= 2
               then Ada.Command_Line.Argument(2)
               else "main");
         begin
            Put_Line(Run_Command("git diff " & Branch));
         end;

      elsif Cmd = "file" then
         if Ada.Command_Line.Argument_Count < 2 then
            Put_Line("ERROR: Usage: review_tool file <file>");
            Ada.Command_Line.Set_Exit_Status(1);
         else
            declare
               Fpath : constant String := Ada.Command_Line.Argument(2);
            begin
               Trace_Utils.Trace_Print("review", "file", Fpath);
               Put_Line("--- Security Check ---");
               Security_Check(Fpath);
               Put_Line("--- Quality Check ---");
               Quality_Check(Fpath);
            end;
         end if;

      elsif Cmd = "security" then
         if Ada.Command_Line.Argument_Count < 2 then
            Put_Line("ERROR: Usage: review_tool security <file>");
            Ada.Command_Line.Set_Exit_Status(1);
         else
            Security_Check(Ada.Command_Line.Argument(2));
         end if;

      elsif Cmd = "quality" then
         if Ada.Command_Line.Argument_Count < 2 then
            Put_Line("ERROR: Usage: review_tool quality <file>");
            Ada.Command_Line.Set_Exit_Status(1);
         else
            Quality_Check(Ada.Command_Line.Argument(2));
         end if;

      else
         Put_Line("ERROR: Unknown command: " & Cmd);
         Ada.Command_Line.Set_Exit_Status(1);
      end if;
   end;
end Review_Tool;
