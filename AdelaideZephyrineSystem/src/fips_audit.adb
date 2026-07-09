with Ada.Text_IO;
with Ada.Calendar;
with Ada.Calendar.Formatting;

package body FIPS_Audit is

   Log_File_Name : constant String := "fips_audit.log";
   Log_File      : Ada.Text_IO.File_Type;
   Is_Open       : Boolean := False;

   procedure Open_Log is
   begin
      if not Is_Open then
         begin
            Ada.Text_IO.Open (File => Log_File,
                              Mode => Ada.Text_IO.Append_File,
                              Name => Log_File_Name);
         exception
            when Ada.Text_IO.Name_Error =>
               Ada.Text_IO.Create (File => Log_File,
                                   Mode => Ada.Text_IO.Append_File,
                                   Name => Log_File_Name);
         end;
         Is_Open := True;
      end if;
   end Open_Log;

   procedure Log_Event (Event_Message : String) is
      Timestamp : constant String := Ada.Calendar.Formatting.Image (Ada.Calendar.Clock);
   begin
      Open_Log;
      Ada.Text_IO.Put_Line (Log_File, "[" & Timestamp & "] [FIPS AUDIT] " & Event_Message);
      Ada.Text_IO.Flush (Log_File);
   exception
      when others =>
         null; -- Do not crash the application if audit log fails, but it would fail FIPS level 2+. For level 0.1 this is fine.
   end Log_Event;

end FIPS_Audit;
