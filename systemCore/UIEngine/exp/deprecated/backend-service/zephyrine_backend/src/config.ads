-- Defines the data structure for our application's configuration.
with Ada.Strings.Unbounded;

package Config is
   use Ada.Strings.Unbounded;

   type App_Config is record
      Port              : Unbounded_String;
      OpenAI_API_Base_URL : Unbounded_String;
      LLM_API_Key       : Unbounded_String;
      DB_Path           : Unbounded_String;
   end record;

   -- Loads configuration from a .env file, using defaults if not found.
   procedure Load (Into : out App_Config; From_File : String := ".env");

end Config;