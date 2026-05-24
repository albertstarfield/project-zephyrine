with Ada.Text_IO; use Ada.Text_IO;
with GNATCOLL.JSON; use GNATCOLL.JSON;

procedure Adelaide_Server_Pkg_Api is
   Val : constant JSON_Value := Read ("{""stream"": false, ""agentic"": true, ""messages"": [{""role"": ""user"", ""content"": ""Hi there zepzep""}]}").Value;
   Is_Streaming : Boolean := False;
   Is_Agentic : Boolean := False;
   Msgs : JSON_Array;
   Last_Msg : JSON_Value;
   Prompt : String (1 .. 100);
begin
   if Has_Field (Val, "stream") then
      Is_Streaming := Get (Val, "stream");
   end if;
   if Has_Field (Val, "agentic") then
      Is_Agentic := Get (Val, "agentic");
   end if;
   if Has_Field (Val, "messages") then
      Msgs := Get (Val, "messages");
      Last_Msg := Get (Msgs, Length (Msgs));
      Put_Line ("Content: " & Get (Last_Msg, "content"));
   end if;
exception
   when E : others => Put_Line ("Exception!");
end Adelaide_Server_Pkg_Api;
