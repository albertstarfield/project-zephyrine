-- connection_manager.ads
with Ada.Containers.Indefinite_Hashed_Maps;
with Ada.Strings.Hash;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with AWS.Net.WebSocket;

package Connection_Manager is

   -- Use the package to make "=" for WebSocket.Object visible
   use AWS.Net.WebSocket;

   type Client_Record is record
      Socket_ID    : AWS.Net.WebSocket.Object;
      User_ID      : Unbounded_String;
      Connected_At : String (1 .. 19); -- YYYY-MM-DD HH:MM:SS
   end record;

   -- Now the instantiation should find "=" for the Key (String) 
   -- and have visibility for the Element type's properties.
   package Client_Maps is new Ada.Containers.Indefinite_Hashed_Maps
     (Key_Type        => String,
      Element_Type    => Client_Record, -- Changed from AWS.Net.WebSocket.Object
      Hash            => Ada.Strings.Hash,
      Equivalent_Keys => "=");

   protected Store is
      procedure Add (User_ID : String; Socket : AWS.Net.WebSocket.Object);
      procedure Remove (User_ID : String);
      function Is_Online (User_ID : String) return Boolean;
      procedure Send_To (User_ID : String; Message : String);
      procedure Broadcast (Message : String);
   private
      Clients : Client_Maps.Map;
   end Store;

end Connection_Manager;