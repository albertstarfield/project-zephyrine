-- File: citation_verifier.adb
-- Citation Verifier - Query Crossref API for paper citations.
-- Note: HTTP requests require external library. Simplified version.

with Ada.Text_IO;
with Ada.Command_Line;
with Ada.Strings;
with Ada.Strings.Unbounded;
with Ada.Processes;
with Trace_Utils;

procedure Citation_Verifier is
   use Ada.Text_IO;
   use Ada.Strings.Unbounded;

begin
   Trace_Utils.Init_Trace;

   if Ada.Command_Line.Argument_Count < 1 then
      Put_Line("Usage: citation_verifier --keywords <query>");
      Put_Line("Note: Requires curl for HTTP requests.");
      Ada.Command_Line.Set_Exit_Status(1);
      return;
   end if;

   --  Parse --keywords argument
   declare
      Keywords : Unbounded_String := Null_Unbounded_String;
      Json_Mode : Boolean := False;
   begin
      for I in 1 .. Ada.Command_Line.Argument_Count loop
         declare
            Arg : constant String := Ada.Command_Line.Argument(I);
         begin
            if Arg = "--keywords" and I < Ada.Command_Line.Argument_Count then
               Keywords :=
                 To_Unbounded_String(Ada.Command_Line.Argument(I + 1));
            elsif Arg = "--json" then
               Json_Mode := True;
            end if;
         end;
      end loop;

      if Length(Keywords) = 0 then
         Put_Line("ERROR: --keywords argument required");
         Ada.Command_Line.Set_Exit_Status(1);
         return;
      end if;

      Trace_Utils.Trace_Print("citation", "query",
        To_String(Keywords));

      --  Use curl to query Crossref API
      declare
         Cmd : constant String :=
           "curl -s 'https://api.crossref.org/works?query=" &
           To_String(Keywords) &
           "&select=DOI,title,author,URL,container-title,issued&rows=1'";
      begin
         begin
            Ada.Processes.Command_Line(
              Command_Line => Cmd,
              Output       => True);
         exception
            when others =>
               Put_Line("ERROR: Failed to query Crossref API");
         end;
      end;
   end;
end Citation_Verifier;
