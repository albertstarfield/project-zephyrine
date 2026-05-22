with Ada.Text_IO; use Ada.Text_IO;
with GNATCOLL.JSON;

procedure Test_JSON is
   Result : GNATCOLL.JSON.Read_Result := GNATCOLL.JSON.Read ("{ 'missing_quotes': val }");
begin
   if Result.Success then
      Put_Line ("SUCCESS");
   else
      Put_Line ("FAIL: " & Result.Error);
   end if;
end Test_JSON;
