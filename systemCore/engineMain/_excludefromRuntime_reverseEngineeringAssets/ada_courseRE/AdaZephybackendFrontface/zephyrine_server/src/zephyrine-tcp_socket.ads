-- File: src/zephyrine-tcp_socket.ads
private with Zephyrine.C_Sockets;

package Zephyrine.TCP_Socket is
   type Socket is limited private;
   Socket_Error : exception;
   procedure Connect (S : out Socket; Host, Port : in String);
   procedure Send (S : in Socket; Data : in String);
   function Receive (S : in Socket; Max_Bytes : Natural := 1024) return String;
   procedure Close (S : in out Socket);
private
   type Socket is limited record
      FD : C_Sockets.Socket_FD := -1;
   end record;
end Zephyrine.TCP_Socket;