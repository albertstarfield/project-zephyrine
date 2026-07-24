pragma SPARK_Mode (Off);
-- c_binding: TPM2 FFI for identity attestation
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package Identity_Manager is

   --  Initialize: Initializes the identity manager and creates the database schema.
   procedure Initialize with Pre => True, Post => True;

   -- Create a new identity. Returns True on success, False if user exists.
   function Register_User (Username, Email, Password : String) return Boolean with Pre => True, Post => True;

   -- Authenticate a user. Returns the Identity Hash on success, empty string on failure.
   function Authenticate_User (Username, Password : String) return String with Pre => True, Post => True;

   -- Helper to compute the 128-bit identity hash
   function Compute_Identity_Hash (Username, Email : String) return String with Pre => True, Post => True;

end Identity_Manager;
