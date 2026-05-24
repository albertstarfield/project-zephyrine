with Ada.Text_IO; use Ada.Text_IO;
with GNATCOLL.JSON; use GNATCOLL.JSON;
procedure Test_JSON is
   Result : Read_Result := Read ("{""stream"": false}");
begin
   if Result.Success then
      declare
         Val : JSON_Value := Result.Value;
         B   : Boolean;
      begin
         B := Get (Val, "stream");
         Put_Line ("OK: " & Boolean'Image(B));
      end;
   end if;
end Test_JSON;
