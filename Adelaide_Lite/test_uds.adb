with GNAT.Sockets; use GNAT.Sockets;
procedure Test_Uds is
   Sock : Socket_Type;
begin
   Create_Socket (Sock, Family_Unix, Socket_Stream);
end Test_Uds;
