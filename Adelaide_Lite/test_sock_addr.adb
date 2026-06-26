with GNAT.Sockets; use GNAT.Sockets;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
procedure Test_Sock_Addr is
   Addr : Sock_Addr_Type (Family_Unix);
begin
   Addr.Name := To_Unbounded_String ("/tmp/adelaide_vad.sock");
end Test_Sock_Addr;
