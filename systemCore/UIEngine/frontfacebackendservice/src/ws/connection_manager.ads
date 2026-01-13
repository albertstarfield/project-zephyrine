with Ada.Containers.Indefinite_Hashed_Maps;
with Ada.Strings.Hash;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with AWS.Net.WebSocket;

package Connection_Manager is

   -- 1. The Data Structure
   type Client_Record is record
      Socket_ID    : AWS.Net.WebSocket.Object;
      User_ID      : Unbounded_String;
      Connected_At : String (1 .. 19);
   end record;

   -- 2. Instantiate the Map OUTSIDE the protected object
   package Client_Maps is new Ada.Containers.Indefinite_Hashed_Maps
     (Key_Type        => String,
      Element_Type    => Client_Record,
      Hash            => Ada.Strings.Hash,
      Equivalent_Keys => "=");

   -- 3. The Thread-Safe Store
   protected Store is
      procedure Add (User_ID : String; Socket : AWS.Net.WebSocket.Object);
      procedure Remove (User_ID : String);
      function Is_Online (User_ID : String) return Boolean;
      procedure Send_To (User_ID : String; Message : String);

   private
      -- The variable lives here, but the type definition is above
      Map : Client_Maps.Map;
   end Store;

end Connection_Manager;