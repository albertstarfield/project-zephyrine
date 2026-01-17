pragma Ada_2022;
with Ada.Text_IO;
with Ada.Calendar;
with Ada.Calendar.Formatting;

package body Connection_Manager is

   protected body Store is

      ---------
      -- Add --
      ---------
      procedure Add (User_ID : String; Socket : AWS.Net.WebSocket.Object) is
         New_Client : Client_Record;
      begin
         New_Client.User_ID      := To_Unbounded_String (User_ID);
         New_Client.Socket_ID    := Socket;
         New_Client.Connected_At := 
           Ada.Calendar.Formatting.Image (Ada.Calendar.Clock)(1 .. 19);

         if Clients.Contains (User_ID) then
            Clients.Replace (User_ID, New_Client);
            Ada.Text_IO.Put_Line ("[WS_Manager] Updated: " & User_ID);
         else
            Clients.Insert (User_ID, New_Client);
            Ada.Text_IO.Put_Line ("[WS_Manager] New: " & User_ID);
         end if;
      end Add;

      ------------
      -- Remove --
      ------------
      procedure Remove (User_ID : String) is
      begin
         if Clients.Contains (User_ID) then
            Clients.Delete (User_ID);
            Ada.Text_IO.Put_Line ("[WS_Manager] Removed: " & User_ID);
         end if;
      end Remove;

      ---------------
      -- Is_Online --
      ---------------
      function Is_Online (User_ID : String) return Boolean is
      begin
         return Clients.Contains (User_ID);
      end Is_Online;

      -------------
      -- Send_To --
      -------------
      procedure Send_To (User_ID : String; Message : String) is
         -- FIX: We need a local variable to act as a 'variable' actual parameter
         Target_Client : Client_Record;
      begin
         if Clients.Contains (User_ID) then
            Target_Client := Clients.Element (User_ID);
            AWS.Net.WebSocket.Send (Target_Client.Socket_ID, Message);
            
            -- If Send modifies the socket state, we might need to 
            -- replace the element back in the map, but usually, 
            -- AWS Sockets are handles where this isn't strictly necessary.
         end if;
      exception
         when others =>
            Ada.Text_IO.Put_Line ("[WS_Manager] Error sending to: " & User_ID);
      end Send_To;

      ---------------
      -- Broadcast --
      ---------------
      procedure Broadcast (Message : String) is
         use Client_Maps;
         C : Cursor := Clients.First;
         -- FIX: Local variable to hold the element during iteration
         Current_Client : Client_Record;
      begin
         while Has_Element (C) loop
            Current_Client := Element (C);
            begin
               AWS.Net.WebSocket.Send (Current_Client.Socket_ID, Message);
            exception
               when others => 
                  Ada.Text_IO.Put_Line ("[WS_Manager] Failed broadcast item");
            end;
            Next (C);
         end loop;
      end Broadcast;

   end Store;

end Connection_Manager;