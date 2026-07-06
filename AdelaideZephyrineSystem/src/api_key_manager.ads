pragma SPARK_Mode (Off);

--  ============================================================================
--  API Key Manager — loads API keys from a plaintext file (one per line) and
--  validates x-api-key headers against them during request dispatch.
--
--  DESIGN
--  ------
--  The key file path is read from the ADELAIDE_API_KEY_FILE environment
--  variable.  Enforcement is toggled by ADELAIDE_API_KEY_ENFORCE (default
--  OFF for backward compatibility with Ollama clients).
--
--  The key file is written by run.py (decrypted from the encrypted store
--  at config/api_keys.enc (local to project) just before spawning the server.
--
--  USAGE
--  -----
--     API_Key_Manager.Initialize;        -- called once at startup
--     if API_Key_Manager.Is_Enforcement_Enabled then
--        if not API_Key_Manager.Validate_API_Key (key) then
--           --  return 401
--        end if;
--     end if;
--  ============================================================================

with Ada.Containers.Ordered_Sets;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package API_Key_Manager is

   --  Load the key file.  Must be called once at server startup.
   --  If ADELAIDE_API_KEY_ENFORCE is not set or is "0", enforcement
   --  remains disabled regardless of the key file contents.
   procedure Initialize;

   --  Return True if API key enforcement is active.
   function Is_Enforcement_Enabled return Boolean;

   --  Validate an x-api-key value against the loaded keys.
   function Validate_API_Key (Key : String) return Boolean;

   --  Number of loaded API keys (0 if file missing / empty).
   function Key_Count return Natural;

private

   --  Ordered set of allowed API keys.
   package Key_Sets is new Ada.Containers.Ordered_Sets
     (Element_Type => Unbounded_String);

   Loaded_Keys : Key_Sets.Set;
   Enforcement : Boolean := False;

end API_Key_Manager;
