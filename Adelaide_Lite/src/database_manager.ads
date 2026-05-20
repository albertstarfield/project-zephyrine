with GNATCOLL.SQL.Exec;
with GNATCOLL.SQL.SQLite;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package Database_Manager is
   pragma Spark_Mode (Off);

   procedure Initialize;

   procedure Remember (User_Input : String; Assistant_Response : String);

   function Recall (Query : String) return String;

   procedure Close;

end Database_Manager;
