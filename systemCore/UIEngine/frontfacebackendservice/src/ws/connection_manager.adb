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
         New_Client.User_ID := To_Unbounded_String (User_ID);
         New_Client.Socket_ID := Socket;
         New_Client.Connected_At := 
            Ada.Calendar.Formatting.Image (Ada.Calendar.Clock);

         -- Insert or Update
         if Map.Contains (User_ID) then
            Map.Replace (User_ID, New_Client);
            Ada.Text_IO.Put_Line ("[WS_Manager] Updated connection for: " & User_ID);
         else
            Map.Insert (User_ID, New_Client);
            Ada.Text_IO.Put_Line ("[WS_Manager] New connection registered: " & User_ID);
         end if;
      end Add;

      ------------
      -- Remove --
      ------------
      procedure Remove (User_ID : String) is
      begin
         if Map.Contains (User_ID) then
            Map.Delete (User_ID);
            Ada.Text_IO.Put_Line ("[WS_Manager] Removed: " & User_ID);
         end if;
      end Remove;

      ---------------
      -- Is_Online --
      ---------------
      function Is_Online (User_ID : String) return Boolean is
      begin
         return Map.Contains (User_ID);
      end Is_Online;

      -------------
      -- Send_To --
      -------------
      procedure Send_To (User_ID : String; Message : String) is
         Client : Client_Record;
      begin
         if Map.Contains (User_ID) then
            Client := Map.Element (User_ID);
            
            -- AWS Thread-Safe Send
            AWS.Net.WebSocket.Send (Client.Socket_ID, Message);
         else
            Ada.Text_IO.Put_Line ("[WS_Manager] Warn: User not found for send: " & User_ID);
         end if;
      exception
         when others =>
            Ada.Text_IO.Put_Line ("[WS_Manager] Error sending to: " & User_ID);
            Remove (User_ID); -- Auto-cleanup on broken pipe
      end Send_To;

   end Store;

end Connection_Manager;